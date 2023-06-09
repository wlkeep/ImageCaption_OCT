import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm


st.title("OCT of Retina Report Generator")

st.markdown("<small>by Lei WANG</small>",unsafe_allow_html=True)
st.markdown("[<small>Github</small>](https://github.com/wlkeep/Image-Caption-of-OCT)",unsafe_allow_html=True)
st.markdown("\nThis app will generate impression part of an OCT report.\nYou can upload an OCT image.")


col1,col2 = st.columns(2)
image = col1.file_uploader("OCT image of Retina",type=['bmp','jpg'])

col1,col2 = st.columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')

@st.cache(allow_output_mutation=True)
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image,model_tokenizer,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        if (image is not None):
            start = time.process_time()  
            image_show = Image.open(image).convert("RGB") #converting to 3 channels
            image_show = np.array(image_show)/255
            st.image([image_show],width=300)
            caption = cm.function1(image_show,model_tokenizer)
            st.markdown(" ### **Impression:**")
            impression = st.empty()
            impression.write(caption[0])
            time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            st.write(time_taken)
            del image
        else:
            st.markdown("## Upload an Image")

def predict_sample(model_tokenizer,folder = './data/image'):
    no_files = len(os.listdir(folder))
    index = np.random.randint(1,no_files)
    image = os.path.join(folder,os.listdir(folder)[index-1])
    predict(image,model_tokenizer,True)

model_tokenizer = create_model()


if test_data:
    predict_sample(model_tokenizer)
else:
    predict(image,model_tokenizer)
