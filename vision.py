import streamlit as st
import cv2
from ultralytics import YOLO
from collections import Counter

# --- 1. CONFIGURAZIONE PAGINA ---
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
st.markdown("<h5 style='text-align: center; color: #aaa;'>Scrivi cosa vuoi trovare e l'AI lo cercherà.</h5>", unsafe_allow_html=True)

# --- 3. SIDEBAR (CONFIGURAZIONE) ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    st.info("👇 Scrivi qui gli oggetti (in inglese), separati da virgola.")
    # Default objects
    default_text = "person, glasses, sunglasses, watch, computer, phone"
    custom_vocab = st.text_area("Oggetti da cercare:", value=default_text, height=100)
    
    confidence = st.slider("Sensibilità", 0.1, 1.0, 0.20, 0.05)
    
    st.divider()
    run = st.toggle("ATTIVA OCCHI AI", value=False)

# --- 4. CARICAMENTO MODELLO (SENZA CACHE PER EVITARE CRASH) ---
# Nota: Abbiamo rimosso @st.cache_resource perché YOLO-World deve 
# resettarsi quando cambiano le classi custom.
def load_model_fresh():
    return YOLO('yolov8s-world.pt')

# --- 5. LOGICA PREPARAZIONE MODELLO ---
if run:
    try:
        # Carichiamo il modello pulito
        model = load_model_fresh()
        
        # Impostiamo le classi SE l'utente ha scritto qualcosa
        if custom_vocab:
            # Pulisce la stringa e crea una lista
            classes_list = [x.strip() for x in custom_vocab.split(',') if x.strip()]
            
            # Imposta le classi custom nel modello
            if classes_list:
                model.set_classes(classes_list)
            
    except Exception as e:
        st.error(f"Errore nel caricamento modello: {e}")
        st.stop()

# --- 6. LAYOUT ---
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Oggetti Trovati")
    stats_placeholder = st.empty()

# --- 7. LOOP WEBCAM ---
if run:
    # Apre la webcam (0 è solitamente quella di default)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("⚠️ Webcam non trovata o occupata.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Errore lettura frame.")
                break
            
            # --- AI DETECTION ---
            # Eseguiamo la predizione
            try:
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                
                # --- STATISTICHE ---
                detected_objects = []
                # Estraiamo i nomi degli oggetti trovati
                for box in results[0].boxes:
                    try:
                        cls_id = int(box.cls[0])
                        # Recuperiamo il nome dalla mappa interna del modello
                        if cls_id in model.names:
                            obj_name = model.names[cls_id]
                        else:
                            obj_name = "Oggetto"
                        detected_objects.append(obj_name.capitalize())
                    except:
                        continue
                
                obj_counts = Counter(detected_objects)
                
                # --- VISUALIZZAZIONE VIDEO ---
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                # Usiamo use_column_width che funziona sia su vecchie che nuove versioni
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # --- VISUALIZZAZIONE STATISTICHE ---
                if obj_counts:
                    stats_html = ""
                    for obj, count in obj_counts.items():
                        # Icone semplici basate sul testo
                        icon = "📦"
                        obj_lower = obj.lower()
                        if "person" in obj_lower: icon = "👤"
                        elif "glass" in obj_lower: icon = "👓"
                        elif "watch" in obj_lower: icon = "⌚"
                        elif "phone" in obj_lower: icon = "📱"
                        elif "comp" in obj_lower: icon = "💻"
                        
                        stats_html += f"""
                        <div class="stat-box">
                            {icon} {obj} <span class="stat-count">{count}</span>
                        </div>
                        """
                    stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                else:
                    stats_placeholder.info("Scansione in corso...")
            
            except Exception as e:
                # Gestione errori durante la predizione (per evitare crash improvvisi)
                print(f"Errore frame: {e}")
                continue
            
        cap.release()
else:
    # Schermata di Standby
    video_placeholder.markdown("""
        <div style='background-color: #111; color: #888; height: 400px; display: flex; align-items: center; justify-content: center; border: 2px dashed #8e44ad; border-radius: 10px;'>
            <h2>Inserisci oggetti nella sidebar e ATTIVA</h2>
        </div>
    """, unsafe_allow_html=True)
    stats_placeholder.empty()