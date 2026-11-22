import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).parent
THRESHOLD = 0.90  # chosen from validation analysis


@st.cache_resource
def load():
    model = tf.keras.models.load_model(HERE / 'agrivigil_mobilenetv2.keras')
    names = json.load(open(HERE / 'class_names.json'))
    return model, names


model, class_names = load()


def nice(name):
    return name.replace('___', ' - ').replace('_', ' ')


st.title('AgriVigil')
st.write('AI plant disease detection with confidence-based expert review.')

file = st.file_uploader('Upload a plant leaf photo', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, width=320)

    x = tf.image.resize(np.array(img), (224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    probs = model.predict(tf.expand_dims(x, 0), verbose=0)[0]

    top3 = np.argsort(probs)[::-1][:3]
    best = int(top3[0])
    conf = float(probs[best])

    if conf >= THRESHOLD:
        st.success(f'Likely: {nice(class_names[best])} ({conf:.0%} confident). This is an AI prediction. Confirm with an expert before applying any treatment.')
    else:
        st.warning(f'Not sure ({conf:.0%} confident). This case should be reviewed by a specialist. Please do not rely on the AI result alone.')

    st.subheader('Top 3 possibilities')
    for i in top3:
        st.write(f'{nice(class_names[i])}: {probs[i]:.0%}')
