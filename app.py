import streamlit as st
from PIL import Image
import torch
import utils

def main():
    model = utils.load_model()

    st.set_page_config(page_title="Task 3 Demo", page_icon="✅", layout="wide")

    st.sidebar.markdown("# ResoluteAI")
    st.sidebar.write("## Task 3: Object Detection and Counting")
    st.sidebar.markdown("---")
    st.sidebar.write("Created by [Suryansh Gupta](https://github.com/suryanshgupta9933)")

    st.title("Task 3: Object Detection and Counting")
    st.markdown("Upload an image and get the detected objects and their count.")
    
    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_thres = st.slider("IOU Threshold", 0.0, 1.0, 0.10, 0.05)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image, img_tensor = utils.preprocess_image(uploaded_file)
        if st.button('Submit'):
            preds = utils.get_predictions(model, img_tensor, conf_thres, iou_thres)
            result_image, count = utils.draw_bounding_boxes(image, preds)
            
            st.markdown(f"### 🔍 Detected Objects: {count}")
            st.markdown("---")
            st.markdown("### 🖼️ Resultant Image:")
            st.image(result_image, caption="Detected Objects")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
