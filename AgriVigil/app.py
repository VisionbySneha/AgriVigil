import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title='AgriVigil', page_icon='🌿', layout='centered')

HERE = Path(__file__).parent
THRESHOLD = 0.90          # chosen from validation analysis
FEEDBACK_URL = ''         # paste your Google Form link here when you have it

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #1B4332, #40916C); padding: 28px 24px;
       border-radius: 16px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 2.2rem; color: #ffffff;}
.hero p {margin: 6px 0 0 0; font-size: 1.05rem; color: #E3F1E8;}
.steps {display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 18px 0;}
.step {background: #E3F1E8; color: #1B4332; border-radius: 999px;
       padding: 6px 14px; font-size: 0.9rem;}
.card {border-radius: 14px; padding: 16px 18px; margin: 12px 0; color: #1B2A22;}
.card.ok {background: #E3F1E8; border-left: 6px solid #2D6A4F;}
.card.warn {background: #FFF4E0; border-left: 6px solid #E09F3E;}
.card h3 {margin: 0 0 4px 0; font-size: 1.3rem;}
.small {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    model = tf.keras.models.load_model(HERE / 'agrivigil_mobilenetv2.keras')
    names = json.load(open(HERE / 'class_names.json'))
    return model, names


model, class_names = load()


def split_label(name):
    crop, _, cond = name.partition('___')
    return crop.replace('_', ' ').strip(), cond.replace('_', ' ').strip()


crops = sorted({split_label(n)[0] for n in class_names})

st.markdown("""
<div class="hero">
  <h1>AgriVigil</h1>
  <p>AI plant disease screening with confidence-based expert review.</p>
</div>
<div class="steps">
  <span class="step">1. Upload a leaf photo</span>
  <span class="step">2. AI checks it</span>
  <span class="step">3. Uncertain cases need an expert</span>
</div>
""", unsafe_allow_html=True)

tab_up, tab_cam = st.tabs(['Upload a photo', 'Use camera'])
with tab_up:
    up = st.file_uploader('Choose a plant leaf photo', type=['jpg', 'jpeg', 'png'])
with tab_cam:
    cam = st.camera_input('Take a photo of the leaf')

file = up if up is not None else cam

with st.expander('Tips and what AgriVigil covers'):
    st.write('For best results, use one leaf in good daylight, close up, with the leaf filling the frame.')
    st.write(f'It knows {len(crops)} crops and {len(class_names)} categories (diseases and healthy leaves). It does not detect pests.')
    st.write('Crops: ' + ', '.join(crops) + '.')

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)

    with st.spinner('Analysing the photo...'):
        x = tf.image.resize(np.array(img), (224, 224))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs = model.predict(tf.expand_dims(x, 0), verbose=0)[0]

    top3 = np.argsort(probs)[::-1][:3]
    best = int(top3[0])
    conf = float(probs[best])
    crop, cond = split_label(class_names[best])
    healthy = 'healthy' in cond.lower()

    if conf >= THRESHOLD:
        title = f'{crop}: looks healthy' if healthy else f'{crop}: {cond}'
        st.markdown(
            f'<div class="card ok"><h3>{title}</h3>'
            f'<div class="small">AI result, {conf:.0%} confident</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card warn"><h3>Not sure about this one</h3>'
            '<div class="small">This photo does not clearly match a supported crop leaf, '
            'or the leaf is hard to read. Try a closer, well-lit photo of a single leaf. '
            'If the problem continues, consult a local agriculture expert or your nearest '
            'Krishi Vigyan Kendra.</div></div>',
            unsafe_allow_html=True)

    st.progress(conf, text=f'Confidence: {conf:.0%}')

    with st.expander('Closest matches (for reference only)'):
        for i in top3:
            c, d = split_label(class_names[i])
            st.progress(float(probs[i]), text=f'{c}: {d} ({probs[i]:.0%})')

    st.caption('This is an AI prediction, not a diagnosis. Confirm with an agriculture expert '
               'before applying any treatment or pesticide.')

    if FEEDBACK_URL:
        st.link_button('Tell us if this was right', FEEDBACK_URL)

st.divider()
st.caption('AgriVigil · B.Tech AI & DS major project, 2026-27 · Aligned with SIH26131 · '
           'Sneha Dikshit and Khushi Prajapati')
