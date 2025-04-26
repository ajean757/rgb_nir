import streamlit as st
from PIL import Image
import os
import glob

DATA_DIR = "./DATA/data_04_06_2025"

warped_rgb_images = glob.glob(f"{DATA_DIR}/warped/*_ir_warped.jpg")
warped_rgb_images.sort()
print(f"Num images: {warped_rgb_images}")

ir_images = []
rgb_images = []

ir_images_rectified = []
rgb_images_rectified = []

overlay_images = []

aerochrome_images = []

for path in warped_rgb_images:
    print(os.path.basename(path).split("_"))
    date, time = os.path.basename(path).split("_")[:2]
    ir_img_path = f"{DATA_DIR}/{date}_{time}_ir.jpg"
    rgb_img_path = f"{DATA_DIR}/{date}_{time}_rgb.jpg"

    ir_img_rect_path = f"{DATA_DIR}/rectified/{date}_{time}_ir_rect.jpg"
    rgb_img_rect_path = f"{DATA_DIR}/rectified/{date}_{time}_rgb_rect.jpg"

    overlay_img_path = f"{DATA_DIR}/overlay/{date}_{time}_overlay.jpg"

    aerochrome_img_path = f"{DATA_DIR}/aerochrome/{date}_{time}_aerochrome.jpg"
    
    ir_images.append(ir_img_path)
    rgb_images.append(rgb_img_path)
    ir_images_rectified.append(ir_img_rect_path)
    rgb_images_rectified.append(rgb_img_rect_path)
    overlay_images.append(overlay_img_path)
    aerochrome_images.append(aerochrome_img_path)



# ir_images = glob.glob(f"{DATA_DIR}/*_ir.jpg") 
# rgb_images = glob.glob(f"{DATA_DIR}/*_rgb.jpg") 
# ir_images.sort()
# rgb_images.sort()

# ir_images_rectified = glob.glob(f"{DATA_DIR}/rectified/*_ir_rect.jpg") 
# rgb_images_rectified = glob.glob(f"{DATA_DIR}/rectified/*_rgb_rect.jpg") 
# ir_images_rectified.sort()
# rgb_images_rectified.sort()

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

img_warped = Image.open(warped_rgb_images[selected_idx])
img_overlay = Image.open(overlay_images[selected_idx])
st.subheader("Warped Images and overlay")
row3 = st.columns(2)
row3[0].image(img_warped, caption="Warped IR Image", use_container_width =True)
row3[1].image(img_overlay, caption="Overlay Image", use_container_width =True)



img_aerochrome = Image.open(aerochrome_images[selected_idx])
st.subheader("Aerochrome (IR False Color)")
row3 = st.columns(1)
row3[0].image(img_aerochrome, caption=f"{aerochrome_images[selected_idx]}", use_container_width =True)

# img_1_warped = Image.open(f"{DATA_DIR}/warped/20250406_170907_rgb.jpg") # hard code for now
# img_overlay = Image.open(f"{DATA_DIR}/overlay/overlay.jpg") # hard code for now
# row3 = st.columns(2)
# row3[0].image(img_1_warped, caption="Warped RGB Image", use_container_width =True)
# row3[1].image(img_overlay, caption="Overlay Image", use_container_width =True)

# To run
# streamlit run viewer.py --server.port 8501 --server.address 0.0.0.0