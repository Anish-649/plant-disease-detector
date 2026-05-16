import streamlit as st
import tensorflow as tf
import numpy as np
import pathlib

# ── Class Labels ──────────────────────────────────────────────────────────────
CLASS_NAME = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = pathlib.Path(__file__).parent / "training_model.keras"
    return tf.keras.models.load_model(str(model_path))

def model_prediction(test_image):
    model = load_plant_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    idx = int(np.argmax(predictions))
    confidence = float(predictions[0][idx]) * 100   # ← confidence in %
    return idx, confidence

def parse_label(raw):
    clean = (raw.replace("(including_sour)", "")
               .replace("(maize)", "")
               .replace(",_bell", "")
               .replace("_(", " ("))
    parts = clean.split("___")
    plant = parts[0].replace("_", " ").strip().title()
    condition = parts[1].replace("_", " ").strip().title() if len(parts) > 1 else raw
    return plant, condition

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load external CSS ─────────────────────────────────────────────────────────
css_path = pathlib.Path(__file__).parent / "style.css"
st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.markdown("""<style>
section[data-testid='stMain'] .block-container,
[data-testid='stMainBlockContainer'] {
    padding-left: 10rem !important;
    padding-right: 10rem !important;
    padding-top: 4rem !important;
    max-width: 100% !important;
}
</style>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div class="header-pill">AI-Powered · 38 Classes · 87K Images</div>
  <div class="header-h1">Plant <span>Disease</span> Detector</div>
  <p class="header-sub">
    Upload a leaf photograph and receive an instant AI-powered diagnosis across 14 crop species —
    powered by a custom CNN trained on 87,000+ augmented plant images.
  </p>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT — Upload + Preview ───────────────────────────────────────────────────
with left:
    st.markdown('<div class="panel-label">① Upload Leaf Image</div>', unsafe_allow_html=True)

    test_image = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if test_image:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">② Preview</div>', unsafe_allow_html=True)
        st.image(test_image, use_container_width=True)
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        analyse = st.button("🔬  Analyse Leaf", type="primary")
    else:
        analyse = False

# ── RIGHT — Result ────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="panel-label">③ Diagnosis Result</div>', unsafe_allow_html=True)

    if not test_image:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🍃</div>
          <div class="empty-title">No Image Yet</div>
          <div class="empty-text">
            Upload a leaf photo on the left<br>then click Analyse Leaf
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif analyse:
        with st.spinner("Running diagnosis…"):
            result_index, confidence = model_prediction(test_image)   # ← unpack both

        raw = CLASS_NAME[result_index]
        plant, condition = parse_label(raw)
        is_healthy = "healthy" in raw.lower()

        # confidence bar color: green if high, amber if medium, red if low
        if confidence >= 75:
            bar_color = "linear-gradient(90deg, #3a6820, #7dba45)"
        elif confidence >= 50:
            bar_color = "linear-gradient(90deg, #7a5a10, #c8a030)"
        else:
            bar_color = "linear-gradient(90deg, #7a2010, #c84030)"

        status_html = (
            '<span class="res-status-healthy">✔ Healthy</span>'
            if is_healthy else
            '<span class="res-status-disease">⚠ Diseased</span>'
        )

        conf_str   = f"{confidence:.1f}"
        status_val = "Healthy ✓" if is_healthy else "Disease Detected"
        html = (
            '<div class="res-card">'
            '<div class="res-top">'
            '<span class="res-top-label">Diagnosis Result</span>'
            + status_html +
            '</div>'
            '<div class="res-body">'
            '<div class="res-plant">' + plant + '</div>'
            '<div class="res-disease">' + condition + '</div>'
            '<div class="res-conf-wrap">'
            '<div class="res-conf-track">'
            '<div class="res-conf-fill" style="width:' + conf_str + '%;background:' + bar_color + '"></div>'
            '</div>'
            '</div>'
            '<div class="res-divider"></div>'
            '<div class="res-meta-grid">'
            '<div class="res-meta-box"><div class="res-meta-key">Crop Species</div><div class="res-meta-val">' + plant + '</div></div>'
            '<div class="res-meta-box"><div class="res-meta-key">Status</div><div class="res-meta-val">' + status_val + '</div></div>'
            '<div class="res-meta-box"><div class="res-meta-key">Condition</div><div class="res-meta-val">' + condition + '</div></div>'
            '<div class="res-meta-box"><div class="res-meta-key">Confidence</div><div class="res-meta-val">' + conf_str + '%</div></div>'
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔬</div>
          <div class="empty-title">Ready to Analyse</div>
          <div class="empty-text">
            Click "Analyse Leaf" to run<br>the AI diagnosis
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-push"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  PlantDisease AI &nbsp;·&nbsp; CNN Model &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; 38 Disease Classes &nbsp;·&nbsp; 14 Crop Species
</div>
""", unsafe_allow_html=True)
