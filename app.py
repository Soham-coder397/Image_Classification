import tensorflow as tf
import numpy as np
import streamlit as st
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: white;
    padding: 18px;
    border-radius: 15px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    font-size: 35px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.result-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<div class='title'>🍎 Fruit & Vegetable Classifier 🥕</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Upload an image and let AI identify the fruit or vegetable.</p>",
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = r"D:\Machine Learning Projects\Image Classification\Image_classify.keras"

model = tf.keras.models.load_model(MODEL_PATH)

img_height = 180
img_width = 180


# ============================================================
# LOAD DATASET
# ============================================================

data_train_path = r"Fruits_Vegitables\train"

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=32
)

data_cat = data_train.class_names


# ============================================================
# FRUIT LIST
# ============================================================

fruits = [
    "Apple",
    "Banana",
    "Cherry",
    "Grapes",
    "Kiwi",
    "Mango",
    "Orange",
    "Papaya",
    "Pineapple",
    "Pomegranate",
    "Strawberry",
    "Watermelon",
    "Coconut"
]


# ============================================================
# IMAGE UPLOADER
# ============================================================

image = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if image is not None:

    # --------------------------------------------------------
    # Two Columns
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 1])

    # --------------------------------------------------------
    # Display Uploaded Image
    # --------------------------------------------------------

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Load Image
    # --------------------------------------------------------

    image_load = tf.keras.utils.load_img(
        image,
        target_size=(img_height, img_width)
    )

    img_arr = tf.keras.utils.img_to_array(image_load)

    img_bat = tf.expand_dims(img_arr, 0)

    # --------------------------------------------------------
    # AI Prediction
    # --------------------------------------------------------

    with st.spinner("🤖 AI is analyzing the image..."):

        time.sleep(2)

        predict = model.predict(
            img_bat,
            verbose=0
        )

        score = tf.nn.softmax(predict)

    # --------------------------------------------------------
    # Get Prediction
    # --------------------------------------------------------

    prediction_index = np.argmax(score)

    prediction = data_cat[prediction_index]

    confidence = float(np.max(score)) * 100

    # --------------------------------------------------------
    # Determine Fruit / Vegetable
    # --------------------------------------------------------

    if prediction in fruits:

        category = "🍎 Fruit"

    else:

        category = "🥕 Vegetable"

    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    with col2:

        st.markdown("## 🔍 Prediction")

        st.success(
            f"### {prediction}"
        )

        st.info(
            f"### Category: {category}"
        )

        st.metric(
            label="🎯 Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(
            confidence / 100
        )

        if confidence >= 80:

            st.success(
                "✅ High confidence prediction"
            )

        elif confidence >= 50:

            st.warning(
                "⚠️ Moderate confidence prediction"
            )

        else:

            st.error(
                "❌ Low confidence prediction"
            )

    # --------------------------------------------------------
    # Detailed Result
    # --------------------------------------------------------

    st.write("")

    st.markdown("---")

    st.markdown("### 📊 Result Summary")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.write("**Detected Object**")

        st.write(f"🍽️ {prediction}")

    with result_col2:

        st.write("**Category**")

        st.write(category)

    with result_col3:

        st.write("**Confidence**")

        st.write(f"{confidence:.2f}%")

    st.write("")

    st.success("🎉 Prediction generated successfully!")


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👆 Please upload a fruit or vegetable image to start prediction."
    )