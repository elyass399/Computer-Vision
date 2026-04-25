# 🌍 AI Vision World

A real-time object detection app powered by **YOLO-World** and **Streamlit**. Point your webcam at anything, type what you want to find, and the AI detects it live.

---

## ✨ Features

- 🔍 **Open-vocabulary detection** — type any object in plain English (e.g. `cat, laptop, coffee cup`)
- 📹 **Live webcam feed** with bounding-box annotations
- 📊 **Real-time statistics** panel showing detected objects and counts
- 🎛️ **Adjustable confidence** slider for fine-tuning sensitivity
- 🌙 **Dark UI** with a clean, minimal layout

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-vision-world.git
cd ai-vision-world
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will automatically download the `yolov8s-world.pt` model (~87 MB) from Ultralytics.

### 3. Run the app

```bash
streamlit run vision.py
```

---

## 📦 Requirements

| Package | Version |
|---|---|
| streamlit | ≥ 1.32 |
| opencv-python | ≥ 4.9 |
| ultralytics | ≥ 8.1 |

Install all at once:

```bash
pip install streamlit opencv-python ultralytics
```

---

## 🖥️ How to Use

1. **Open the sidebar** (top-left arrow if on mobile)
2. **Type objects** you want to detect, separated by commas — in English (e.g. `person, phone, bottle, dog`)
3. **Adjust the sensitivity** slider (lower = more detections, higher = stricter)
4. **Toggle "ATTIVA OCCHI AI"** to start the live feed
5. Watch detections appear in the video and the stats panel on the right

---

## 📁 Project Structure

```
ai-vision-world/
├── vision.py          # Main Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

The `yolov8s-world.pt` model file is downloaded automatically on first run and cached locally.

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Objects to detect | `person, glasses, sunglasses, watch, computer, phone` | Comma-separated list in English |
| Sensitivity | `0.20` | Confidence threshold (0.1–1.0) |

---

## 🔧 Troubleshooting

**Webcam not found**
- Make sure no other app is using the webcam
- On Linux, try `ls /dev/video*` to check available devices; change `cv2.VideoCapture(0)` to `1` or `2` if needed

**Model download fails**
- Ensure you have internet access on first run
- Manually download from [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place `yolov8s-world.pt` in the project root

**Low FPS / lag**
- Raise the confidence threshold to reduce processing load
- Close other resource-heavy applications
- Use a smaller model variant if available

**`use_column_width` deprecation warning**
- This is a cosmetic warning from newer Streamlit versions; it does not affect functionality

---

## 📄 License

MIT — free to use and modify.
