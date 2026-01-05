import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
# import cv2

from DigitRecognitionSystem import NeuralNetwork

if 'model' not in st.session_state:
    st.session_state.model = NeuralNetwork()
    try:
        st.session_state.model.load_model()
    except FileNotFoundError:
        pass
st.title("Digit Recognition System(Inaccurate on real-data[fuck], now what does that mean? it means we overfitted lads fuck yeah)")
st.write("Draw a digit (0-9) in the canvas below")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Drawing Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=35,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Controls")

    predict_btn = st.button("Predict Digit", use_container_width=True)

    st.markdown("---")
    st.subheader("Prediction Result")

    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()

if predict_btn and canvas_result.image_data is not None:
    img_data = canvas_result.image_data

    img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

    img = img.convert('L')

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 784)

    prediction = st.session_state.model.predict(img_array)

    probabilities = st.session_state.model.forward_pass(img_array)[0]
    confidence = probabilities[prediction] * 100

    with col2:
        prediction_placeholder.markdown(
            f"<h1 style='text-align: center; color: #4CAF50;'>{prediction}</h1>",
            unsafe_allow_html=True
        )
        confidence_placeholder.markdown(
            f"<p style='text-align: center;'>Confidence: {confidence:.2f}%</p>",
            unsafe_allow_html=True
        )

        st.subheader("Probability Distribution")
        prob_df = {
            'Digit': list(range(10)),
            'Probability': [f"{p * 100:.2f}%" for p in probabilities]
        }
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
