import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

st.title("28x28 Drawing Canvas")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Red channel lang
    img_gray = img[:, :, 0]

    img_pil = Image.fromarray(img_gray.astype('uint8'))
    img_28x28 = img_pil.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img_28x28)

    st.write("28x28 Output:")
    st.image(img_28x28, width=140)
