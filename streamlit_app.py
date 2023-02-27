import streamlit as st
import base64
from cogniflow_utils import cogniflow_request_object
import cv2
import numpy as np

st.set_page_config(
    page_title="Cogniflow Image Demo",
    page_icon="https://uploads-ssl.webflow.com/60510407e7726b268293da1c/60ca08f7a2abc9c7c79c4dac_logo_ico256x256.png",
)

def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.title('🔎 Object Detection and Counting')
st.markdown("Powered by [Cogniflow](https://www.cogniflow.ai)")

model = st.secrets["model_url"]
api_key = st.secrets["api_key"]

if not "image/object-detection/detect/" in model:
    st.error("Error validating model url. Please make sure you want to use an object detection model")
    st.stop()

file = st.file_uploader("Upload a picture", type=['jpg', 'png', 'jpeg'])

st.session_state['enableBtn'] = not (file is not None and model != "" and api_key != "")

click = st.button("✨ Detect objects", disabled=st.session_state.enableBtn)

if click:
    if file is not None and model != "" and api_key != "":    
        image_format = file.type.replace("image/", "")
        bytes_data = file.getvalue()
        image_b64 = base64.b64encode(bytes_data).decode()

        with st.spinner("Predicting..."):
            result = cogniflow_request_object(model, api_key, image_b64, image_format)

        boxes = result['result']

        categories = {}
        for box in boxes:
            c = box['category']
            if c in categories:
                categories[c] += 1
            else:
                categories[c] = 1

        categories_list = list(categories)

        colors = np.random.randint(0, 255, size=(len(categories_list), 3), dtype='uint8')

        img_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)        
        imgResult = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        for box in boxes:            
            cat = box['category']
            index_color = categories_list.index(cat) 
            color = colors[index_color]
            bgr = [int(c) for c in color]
            cv2.rectangle(imgResult, (int(box['x_left_top']), int(box['y_left_top'])), (int(box['x_right_bottom']), int(box['y_right_bottom'])), bgr, 2)
            cv2.putText(imgResult, cat, (int(box['x_left_top']), int(box['y_left_top']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

        imageRGB = cv2.cvtColor(imgResult , cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2, gap="large")

        col1.write("Image:")
        col1.image(imageRGB)
        col2.write("Objects Counted:")
        col2.write(categories)
