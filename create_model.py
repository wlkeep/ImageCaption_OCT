import joblib
import os
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Input, Embedding, LSTM,Dot,Reshape,Concatenate,BatchNormalization, GlobalMaxPooling2D, Dropout, Add, MaxPooling2D, GRU, AveragePooling2D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import cv2
from nltk.translate.bleu_score import sentence_bleu

vgg16_weights = "./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def create_vgg16(vgg16_weights = vgg16_weights):
  """
  chexnet_weights: weights value in .h5 format of chexnet
  creates a chexnet model with preloaded weights present in chexnet_weights file
  """
  model = tf.keras.applications.VGG16(include_top=False) #importing densenet the last layer will be a relu activation layer

  vgg16 = tf.keras.Model(inputs = model.input,outputs = model.output)
  vgg16.load_weights(vgg16_weights)
  vgg16 = tf.keras.Model(inputs = model.input,outputs = vgg16.layers[-1].output)  #we will be taking the penultimate layer (second last layer here it is global avgpooling)
  return vgg16


class Image_encoder(tf.keras.layers.Layer):
  """
  This layer will output image backbone features after passing it through chexnet
  here chexnet will be not be trainable
  """
  def __init__(self,
               name = "image_encoder_block"
               ):
    super().__init__()
    self.vgg16 = create_vgg16()
    self.vgg16.trainable = False
    #for i in range(10): #the last 10 layers of chexnet will be trained
     #self.vgg16.layers[-i].trainable = True
    
  def call(self,data):
    op = self.vgg16(data)
    return op


def encoder(image,dense_dim = dense_dim,dropout_rate = dropout_rate):
  """
  Takes image1,image2
  gets the final encoded vector of these
  """
  #image1
  im_encoder = Image_encoder()
  bkfeat = im_encoder(image) #shape: (None,9,1024)
  bk_featnorm = BatchNormalization()(bkfeat)
  bk_featfinal = GlobalAveragePooling2D()(bk_featnorm)
  bk_dense = Dense(dense_dim,name = 'bkdense',activation = 'relu') #shape: (None,9,512)
  bk_feat = bk_dense(bk_featfinal)

  bn = BatchNormalization(name = "encoder_batch_norm")(bk_feat) 
  dropout = Dropout(dropout_rate,name = "encoder_dropout")(bn)
  return dropout


class global_attention(tf.keras.layers.Layer):
  """
  calculate global attention
  """
  def __init__(self,dense_dim = dense_dim):
    super().__init__()
    # Intialize variables needed for Concat score function here
    self.W1 = Dense(units = dense_dim) #weight matrix of shape enc_units*dense_dim
    self.W2 = Dense(units = dense_dim) #weight matrix of shape dec_units*dense_dim
    self.V = Dense(units = 1) #weight matrix of shape dense_dim*1 
      #op (None,98,1)


  def call(self,encoder_output,decoder_h): #here the encoded output will be the concatted image bk features shape: (None,98,dense_dim)
    decoder_h = tf.expand_dims(decoder_h,axis=1) #shape: (None,1,dense_dim)
    tanh_input = self.W1(encoder_output) + self.W2(decoder_h) #ouput_shape: batch_size*98*dense_dim
    tanh_output =  tf.nn.tanh(tanh_input)
    attention_weights = tf.nn.softmax(self.V(tanh_output),axis=1) #shape= batch_size*98*1 getting attention alphas
    op = attention_weights*encoder_output#op_shape: batch_size*98*dense_dim  multiply all aplhas with corresponding context vector
    context_vector = tf.reduce_sum(op,axis=1) #summing all context vector over the time period ie input length, output_shape: batch_size*dense_dim


    return context_vector,attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
  """
  decodes a single token
  """
  def __init__(self,vocab_size = vocab_size, embedding_dim = embedding_dim, max_pad = max_pad, dense_dim = dense_dim ,name = "onestepdecoder"):
    # Initialize decoder embedding layer, LSTM and any other objects needed
    super().__init__()
    self.dense_dim = dense_dim
    self.embedding = Embedding(input_dim = vocab_size+1,
                                output_dim = embedding_dim,
                                input_length=max_pad,
                                weights = [embedding_matrix],
                                mask_zero=True, 
                                name = 'onestepdecoder_embedding'
                              )
    self.LSTM = GRU(units=self.dense_dim,
                    # return_sequences=True,
                    return_state=True,
                    name = 'onestepdecoder_LSTM'
                    )
    self.attention = global_attention(dense_dim = dense_dim)
    self.concat = Concatenate(axis=-1)
    self.dense = Dense(dense_dim,name = 'onestepdecoder_embedding_dense',activation = 'relu')
    self.final = Dense(vocab_size+1,activation='softmax')
    self.add =Add()
  @tf.function
  def call(self,input_to_decoder, encoder_output, decoder_h):#,decoder_c):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B

      here state_h,state_c are decoder states
    '''
    embedding_op = self.embedding(input_to_decoder) #output shape = batch_size*1*embedding_shape (only 1 token)
    
    context_vector,attention_weights = self.attention(encoder_output,decoder_h) #passing hidden state h of decoder and encoder output
    #context_vector shape: batch_size*dense_dim we need to add time dimension
    context_vector_time_axis = tf.expand_dims(context_vector,axis=1)
    #now we will combine attention output context vector with next word input to the lstm here we will be teacher forcing
    concat_input = self.concat([context_vector_time_axis,embedding_op])#output dimension = batch_size*input_length(here it is 1)*(dense_dim+embedding_dim)
    
    output,decoder_h = self.LSTM(concat_input,initial_state = decoder_h)
    #output shape = batch*1*dense_dim and decoder_h,decoder_c has shape = batch*dense_dim
    #we need to remove the time axis from this decoder_output
    

    output = self.final(output)#shape = batch_size*decoder vocab size
    return output,decoder_h,attention_weights


class decoder(tf.keras.Model):
  """
  Decodes the encoder output and caption
  """
  def __init__(self,max_pad = max_pad, embedding_dim = embedding_dim,dense_dim = dense_dim,score_fun='general',batch_size = batch_size,vocab_size = vocab_size):
    super().__init__()
    self.onestepdecoder = One_Step_Decoder(vocab_size = vocab_size, embedding_dim = embedding_dim, max_pad = max_pad, dense_dim = dense_dim)
    self.output_array = tf.TensorArray(tf.float32,size=max_pad)
    self.max_pad = max_pad
    self.batch_size = batch_size
    self.dense_dim =dense_dim
    
  @tf.function
  def call(self,encoder_output,caption):#,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
    decoder_h, decoder_c = tf.zeros_like(encoder_output), tf.zeros_like(encoder_output) #decoder_h, decoder_c
    output_array = tf.TensorArray(tf.float32,size=max_pad)
    for timestep in range(self.max_pad): #iterating through all timesteps ie through max_pad
      output,decoder_h,attention_weights = self.onestepdecoder(caption[:,timestep:timestep+1], encoder_output, decoder_h)
      output_array = output_array.write(timestep,output) #timestep*batch_size*vocab_size

    self.output_array = tf.transpose(output_array.stack(),[1,0,2]) #.stack :Return the values in the TensorArray as a stacked Tensor.)
        #shape output_array: (batch_size,max_pad,vocab_size)
    return self.output_array


def create_model():
  """
  creates the best model ie the attention model
  and returns the model after loading the weights
  and also the tokenizer
  """
  #hyperparameters
  input_size = (224,224)
  tokenizer = joblib.load('./tokenizer.pkl')
  max_pad = 15
  batch_size = 8
  vocab_size = len(tokenizer.word_index)
  embedding_dim = 200
  dense_dim = 512
  lstm_units = dense_dim
  dropout_rate = 0.4


  tf.keras.backend.clear_session()
  image = Input(shape = (input_size + (3,))) #shape = 224,224,3

  caption = Input(shape = (max_pad,))

  encoder_output = encoder(image,dense_dim,dropout_rate) #shape: (None,28,512)

  output = decoder(max_pad, embedding_dim,dense_dim,batch_size ,vocab_size)(encoder_output,caption)
  model = tf.keras.Model(inputs = [image,caption], outputs = output)
  model_filename = './Encoder_Decoder_global_attention.h5'
  model_save = model_filename
  model.load_weights(model_save)

  return model,tokenizer


def greedy_search_predict(image,model = model1):
  """
  Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
  """
  image = cv2.imread(image)/255 
  image = tf.expand_dims(cv2.resize(image,input_size,interpolation = cv2.INTER_NEAREST),axis=0) #introduce batch and resize
  image = model.get_layer('image_encoder')(image)
  image_norm = model.get_layer('batch_normalization')(image)
  image_global = model.get_layer('global_average_pooling2d')(image_norm)
  image_dense = model.get_layer('bkdense')(image_global)
  enc_op = model.get_layer('encoder_dropout')(image_dense) #this is the output from encoder
  print(enc_op.shape)


  decoder_h,decoder_c = tf.zeros_like(enc_op),tf.zeros_like(enc_op)
  a = []
  pred = []
  for i in range(max_pad):
    if i==0: #if first word
      caption = np.array(tokenizer.texts_to_sequences(['<CLS>'])) #shape: (1,1)
    output,decoder_h,attention_weights = model.get_layer('decoder').onestepdecoder(caption,enc_op,decoder_h)#,decoder_c) decoder_c,

    #prediction
    max_prob = tf.argmax(output,axis=-1)  #tf.Tensor of shape = (1,1)
    caption = np.array([max_prob]) #will be sent to onstepdecoder for next iteration
    if max_prob==np.squeeze(tokenizer.texts_to_sequences([''])): 
      break;
    else:
      a.append(tf.squeeze(max_prob).numpy())
  return tokenizer.sequences_to_texts([a])[0].split("<END>")[0].strip() #here output would be 1,1 so subscripting to open the array


def get_bleu(reference,prediction):
  """
  Given a reference and prediction string, outputs the 1-gram,2-gram,3-gram and 4-gram bleu scores
  """
  reference = [reference.split()] #should be in an array (cos of multiple references can be there here only 1)
  prediction = prediction.split()
  bleu1 = sentence_bleu(reference,prediction,weights = (1,0,0,0))
  bleu2 = sentence_bleu(reference,prediction,weights = (0.5,0.5,0,0))
  bleu3 = sentence_bleu(reference,prediction,weights = (0.33,0.33,0.33,0))
  bleu4 = sentence_bleu(reference,prediction,weights = (0.25,0.25,0.25,0.25))

  return bleu1,bleu2,bleu3,bleu4


def predict1(image,model_tokenizer = None):
  """given image1 and image 2 filepaths returns the predicted caption,
  the model_tokenizer will contain stored model_weights and tokenizer 
  """
#   try:
#     image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255 
#     image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255
#   except:
#     return print("Must be an image")

  if model_tokenizer == None:
    model,tokenizer = create_model()
  else:
    model,tokenizer = model_tokenizer[0],model_tokenizer[1]
  predicted_caption = greedy_search_predict(image,model,tokenizer)

  return predicted_caption


def predict2(true_caption, image,model_tokenizer = None):
  """given image1 and image 2 filepaths and the true_caption
   returns the mean of cumulative ngram bleu scores where n=1,2,3,4,
  the model_tokenizer will contain stored model_weights and tokenizer 
  """
  try:
    image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255 
    image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255
  except:
    return print("Must be an image")

  if model_tokenizer == None:
    model,tokenizer = create_model()
  else:
    model,tokenizer = model_tokenizer[0],model_tokenizer[1]
  predicted_caption = greedy_search_predict(image,model,tokenizer)

  _ = get_bleu(true_caption,predicted_caption)
  _ = list(_)
  return pd.DataFrame([_],columns = ['bleu1','bleu2','bleu3','bleu4'])


def function1(image,model_tokenizer = None):
  """
  here image1 and image2 will be a list of image
  filepaths and outputs the resulting captions as a list
  """
  if model_tokenizer is None:
    model_tokenizer = list(create_model())
  predicted_caption = []
  for i in zip(image):
    caption = predict1(i,model_tokenizer)
    predicted_caption.append(caption)

  return predicted_caption

def function2(true_caption,image):
  """
  here true_caption,image1 and image2 will be a list of true_captions and image
  filepaths and outputs the resulting bleu_scores
  as a dataframe
  """
  model_tokenizer = list(create_model())
  predicted = pd.DataFrame(columns = ['bleu1','bleu2','bleu3','bleu4'])
  for c,i in zip(true_caption,image):
    caption = predict2(c,i,model_tokenizer)
    predicted = predicted.append(caption,ignore_index = True)

  return predicted