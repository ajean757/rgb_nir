import streamlit as st
from PIL import Image
import os
import glob

DATA_DIR = "./DATA/data_04_06_2025"
ir_images = glob.glob(f"{DATA_DIR}/*_ir.jpg") 
rgb_images = glob.glob(f"{DATA_DIR}/*_rgb.jpg") 
ir_images.sort()
rgb_images.sort()

ir_images_rectified = glob.glob(f"{DATA_DIR}/rectified/*_ir_rect.jpg") 
rgb_images_rectified = glob.glob(f"{DATA_DIR}/rectified/*_rgb_rect.jpg") 
ir_images_rectified.sort()
rgb_images_rectified.sort()

st.title("Stereo Pipeline Visualizer")

selected_idx = st.slider("Image index", 0, len(ir_images)-1)
img1 = Image.open(rgb_images[selected_idx])
img2 = Image.open(ir_images[selected_idx])

img1_rect = Image.open(rgb_images_rectified[selected_idx])
img2_rect = Image.open(ir_images_rectified[selected_idx])

st.subheader("Original Images")
row1 = st.columns(2)
row1[0].image(img1, caption="RGB Image", use_container_width =True)
row1[1].image(img2, caption="IR Image", use_container_width =True)

st.subheader("Rectified Images")
row2 = st.columns(2)
row2[0].image(img1_rect, caption="Rectified RGB Image", use_container_width =True)
row2[1].image(img2_rect, caption="Rectified IR Image", use_container_width =True)


# To run
# streamlit run viewer.py --server.port 8501 --server.address 0.0.0.0