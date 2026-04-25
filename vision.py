import streamlit as st
import cv2
from ultralytics import YOLO
from collections import Counter

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="AI Vision World",
    page_icon="🌍",
    layout="wide"
)

# --- 2. CSS ---
st.markdown("""
<style>
    h1 { color: #8e44ad; text-align: center; }
    .stApp { background-color: #0e1117; }
    .stat-box {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #8e44ad;
        margin-bottom: 10px;
        color: white;
        font-size: 18px;
    }
    .stat-count { font-weight: bold; color: #8e44ad; float: right; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 AI Vision World: Cerca Qualsiasi Cosa")
st.markdown(
    "<h5 style='text-align: center; color: #aaa;'>Scrivi cosa vuoi trovare e l'AI lo cercherà.</h5>",
    unsafe_allow_html=True
)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    st.info("👇 Scrivi qui gli oggetti (in inglese), separati da virgola.")

    default_text = "person, glasses, sunglasses, watch, computer, phone"
    custom_vocab = st.text_area("Oggetti da cercare:", value=default_text, height=100)
    confidence = st.slider("Sensibilità", 0.1, 1.0, 0.20, 0.05)
    st.divider()
    run = st.toggle("ATTIVA OCCHI AI", value=False)


# --- 4. CACHED MODEL LOADER ---
# Cache by the exact set of classes so the model only reloads when classes change.
@st.cache_resource
def load_model(classes_key: str):
    """Load YOLO-World and set custom classes. Cached per unique class list."""
    model = YOLO("yolov8s-world.pt")
    classes_list = [c.strip() for c in classes_key.split(",") if c.strip()]
    if classes_list:
        model.set_classes(classes_list)
    return model


# --- 5. LAYOUT ---
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Oggetti Trovati")
    stats_placeholder = st.empty()


# --- 6. ICON HELPER ---
ICON_MAP = {
    "person": "👤",
    "glass": "👓",    # matches "glasses" and "sunglasses"
    "watch": "⌚",
    "phone": "📱",
    "comp": "💻",
}

def get_icon(label: str) -> str:
    label_lower = label.lower()
    for keyword, icon in ICON_MAP.items():
        if keyword in label_lower:
            return icon
    return "📦"


# --- 7. WEBCAM LOOP ---
if run:
    # Normalise the class string so the cache key is stable
    classes_key = ", ".join(
        c.strip() for c in custom_vocab.split(",") if c.strip()
    )

    try:
        model = load_model(classes_key)
    except Exception as e:
        st.error(f"Errore nel caricamento modello: {e}")
        st.stop()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("⚠️ Webcam non trovata o occupata.")
    else:
        # Place a stop button so the user can end the loop without toggling
        stop = st.button("⏹ Ferma", use_container_width=True)

        while not stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("Errore lettura frame.")
                break

            try:
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()

                # Collect detected labels
                detected_objects = []
                for box in results[0].boxes:
                    try:
                        cls_id = int(box.cls[0])
                        label = model.names.get(cls_id, "Oggetto")
                        detected_objects.append(label.capitalize())
                    except Exception:
                        continue

                obj_counts = Counter(detected_objects)

                # Display video — use_container_width replaces deprecated use_column_width
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Display stats
                if obj_counts:
                    stats_html = "".join(
                        f'<div class="stat-box">'
                        f'{get_icon(obj)} {obj} <span class="stat-count">{count}</span>'
                        f'</div>'
                        for obj, count in obj_counts.items()
                    )
                    stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                else:
                    stats_placeholder.info("Scansione in corso...")

            except Exception as e:
                # Log frame errors without crashing the loop
                print(f"Errore frame: {e}")
                continue

        cap.release()

else:
    # Standby screen
    video_placeholder.markdown("""
        <div style='background-color: #111; color: #888; height: 400px;
                    display: flex; align-items: center; justify-content: center;
                    border: 2px dashed #8e44ad; border-radius: 10px;'>
            <h2>Inserisci oggetti nella sidebar e ATTIVA</h2>
        </div>
    """, unsafe_allow_html=True)
    stats_placeholder.empty()
