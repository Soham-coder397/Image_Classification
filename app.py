import tensorflow as tf
import numpy as np
import streamlit as st
import time

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:white;
    padding:18px;
    border-radius:15px;
    background:linear-gradient(90deg,#ff512f,#dd2476);
    font-size:35px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-box{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<div class='title'>Fruit & Vegetable Classifier</div>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Upload an image and let AI identify the fruit or vegetable.</p>", unsafe_allow_html=True)

st.write("")

# ---------- Load Model ----------
model = tf.keras.models.load_model(r'D:\Python Projects\Machine Learning\Image Classification\Image_classify.keras')

img_height = 180
img_width = 180

# ---------- Dataset ----------
data_train_path = r'Fruits_Vegitables\train'

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=32
)

data_cat = data_train.class_names

# ---------- Upload Image ----------
image = st.file_uploader("Upload an Image",type=["jpg", "jpeg", "png"])

if image is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    image_load = tf.keras.utils.load_img(
        image,
        target_size=(img_height, img_width)
    )

    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat, verbose=0)

    score = tf.nn.softmax(predict)

    prediction = data_cat[np.argmax(score)]
    confidence = float(np.max(score)) * 100

    with st.spinner("AI is analyzing the image..."):

        time.sleep(2)

        predict = model.predict(img_bat, verbose=0)

        score = tf.nn.softmax(predict)

    with col2:

        st.markdown("## Prediction")

        st.success(f"**{prediction}**")

        st.metric(label="Confidence",value=f"{confidence:.2f}%")

        st.progress(confidence / 100)

        st.info("Prediction generated successfully")