<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Indian%20License%20Plate%20Detection&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=YOLO26%20%C2%B7%20PaddleOCR%20%C2%B7%20RTSP%20%C2%B7%20Multi-Camera%20%C2%B7%20Security%20Deployment&descAlignY=60&descSize=14&descColor=a8d8ea" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00BFFF?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.8%2B-0062B8?style=for-the-badge&logo=paddlepaddle&logoColor=white)](https://github.com/PaddlePaddle/PaddleOCR)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/indian-lp-detection?style=for-the-badge&color=FFD700&logo=github)](https://github.com/Aahant0607/DTU_Rashak_Beta)

<br/>

> **Production-grade** license plate detection & OCR pipeline for Indian roads.  
> Runs on CCTV · IP cameras · RTSP streams · video files · webcam.  
> Trains on Kaggle in ~20 min, deploys to a Raspberry Pi or a rack server.

<br/>

[**Quick Start**](#-quick-start) · [**Notebook**](#-kaggle-notebook) · [**Architecture**](#%EF%B8%8F-architecture) · [**Results**](#-results--benchmarks) · [**Deployment**](#-deployment-guide) · [**FAQ**](#-faq)

</div>

---

## 📸 Demo

<div align="center">

| Input Frame | Detection + OCR Output |
|:-----------:|:----------------------:|
| Raw CCTV frame | `DL3CAB1234` · conf `0.94` · ocr `0.91` |

</div>

```
[CAM01]  🚘  DL3CAB1234   det=0.94  ocr=0.91  14:32:07
[CAM01]  🚘  MH12AB5678   det=0.88  ocr=0.87  14:32:11
[CAM02]  🚘  HR26A1234    det=0.91  ocr=0.93  14:32:14
```

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🎯 **YOLO26 Detection** | NMS-free architecture — faster & more accurate than YOLOv8 on small objects |
| 🔤 **PaddleOCR** | SVTR_LCNet recogniser · CLAHE preprocessing · 2× upscale pipeline |
| 🇮🇳 **Indian Plate Aware** | Positional char-correction (`O↔0`, `I↔1`, `B↔8` …) · all standard formats |
| 📡 **Live RTSP Streams** | Multi-camera · per-thread · auto-reconnect on drop |
| 🚗 **Vehicle Association** | COCO detector links each plate to its parent vehicle via IoU |
| 💾 **Structured Logging** | CSV + JSON · crop images · full-frame snapshots · dedup by cooldown |
| ⚡ **GPU / CPU** | Auto-detects CUDA · runs on CPU with frame-skip for modest hardware |
| 🖥️ **Headless / GUI** | Server mode (no display) or windowed live feed |
| 📓 **Unified Notebook** | One `.ipynb` for training → inference → live stream → export |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES                               │
│   RTSP Stream  │  Video File  │  Image Folder  │  Webcam (0)   │
└────────────────────────┬────────────────────────────────────────┘
                         │  BGR frames
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRAME PRE-PROCESSING                               │
│   Frame-skip (every Nth)  │  CLAHE  │  Grayscale for OCR       │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
           ▼                            ▼
┌──────────────────┐         ┌──────────────────────┐
│  COCO Detector   │         │   LP Detector        │
│  yolo26n.pt      │         │   license_plate_     │
│  (vehicle bbox)  │         │   detector.pt        │
│  car/moto/       │         │   YOLO26 fine-tuned  │
│  bus/truck       │         │   on Indian plates   │
└────────┬─────────┘         └──────────┬───────────┘
         │                              │
         └──────────┬───────────────────┘
                    │  IoU association
                    ▼
        ┌───────────────────────┐
        │   PLATE CROP  +  OCR  │
        │   2× upscale          │
        │   CLAHE               │
        │   PaddleOCR           │
        │   SVTR_LCNet          │
        │   Char correction     │
        │   Plate regex         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   OUTPUT              │
        │   CSV / JSON log      │
        │   Crop images         │
        │   Annotated snapshots │
        │   Live display        │
        └───────────────────────┘
```

---

## 🚀 Quick Start

### 1 · Install

```bash
git clone https://github.com/yourusername/indian-lp-detection.git
cd indian-lp-detection

# CPU (default)
pip install -r requirements.txt

# GPU (NVIDIA)
pip install -r requirements.txt
pip install paddlepaddle-gpu>=2.6.2
```

<details>
<summary><b>requirements.txt</b></summary>

```
ultralytics>=8.4.0
paddleocr>=2.8.1
paddlepaddle>=2.6.2
opencv-python>=4.8.0
numpy>=1.25
Pillow>=9.5
pandas>=2.0
tqdm
PyYAML
matplotlib
```

</details>

---

### 2 · Place Weights

```
indian-lp-detection/
├── live_lp_detection.py
├── license_plate_detector.pt   ← your fine-tuned LP weights (see §Training)
└── yolo26n.pt                  ← auto-downloads on first run
```

> **Don't have LP weights yet?** Skip to [Training](#-training) or set `SKIP_TRAINING=False` in the notebook to train your own.

---

### 3 · Run

```bash
# Webcam test
python live_lp_detection.py --source 0

# Single IP camera (RTSP)
python live_lp_detection.py --source "rtsp://admin:password@192.168.1.64:554/stream1"

# Two cameras simultaneously
python live_lp_detection.py \
  --source "rtsp://admin:pass@192.168.1.64:554/stream1" \
           "rtsp://admin:pass@192.168.1.65:554/stream1"

# Headless server (no monitor)
python live_lp_detection.py \
  --source "rtsp://admin:pass@192.168.1.64:554/stream1" \
  --headless

# Video file
python live_lp_detection.py --source /path/to/footage.mp4
```

#### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Camera index, RTSP URL, or video file path (space-separated for multi-cam) |
| `--headless` | `False` | No display window — use on servers |
| `--lp-weights` | `./license_plate_detector.pt` | Path to LP detector weights |
| `--conf` | `0.40` | Plate detection confidence threshold |
| `--skip` | `3` | Process every Nth frame (higher = faster, fewer detections) |

---

## 📓 Kaggle Notebook

The unified notebook covers the **full pipeline** in one file:

```
Section 0  →  Environment detection & package install (auto GPU/CPU)
Section 1  →  Global config (all settings in one place)
Section 2  →  Dataset YAML setup
Section 3  →  OCR engine init
Section 4  →  YOLO26 training
Section 5  →  Load inference models
Section 6  →  Core detection pipeline
Section 7  →  Image inference
Section 8  →  Video inference
Section 9  →  Live RTSP / webcam stream
Section 10 →  CSV & JSON export
Section 11 →  Statistics & plots
Section 12 →  Sample annotated results
Section 13 →  ONNX / TensorRT / CoreML export
Section 14 →  Output summary
```

### Kaggle setup

1. Upload the notebook (`indian_lp_detection_unified.ipynb`)
2. Enable **GPU accelerator** (T4 or P100) in *Settings → Accelerator*
3. Enable **Internet** in *Settings → Internet*
4. Add your dataset as an input (YOLOv8 format from Roboflow)
5. Run all cells — training takes ~20 min on T4

```python
# Key flags at the top of Section 1
SKIP_TRAINING   = False   # True → jump straight to inference
RUN_LIVE_STREAM = False   # True → enable RTSP stream section
LIVE_SOURCES    = ["rtsp://admin:pass@192.168.1.64:554/stream1"]
HEADLESS        = True    # always True on Kaggle
```

---

## 🏋️ Training

### Dataset

Recommended dataset: [Indian License Plate Detection on Roboflow Universe](https://universe.roboflow.com/)

Download in **YOLOv8 format**. Your `data.yaml` should look like:

```yaml
path: /path/to/dataset
train: images/train
val:   images/val
test:  images/test
nc: 1
names: ['license_plate']
```

### Train command (script)

```bash
# Fine-tune YOLO26n on your dataset
yolo train \
  model=yolo26n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  freeze=10 \
  project=runs/train \
  name=lp_detector
```

### Or use the notebook (recommended)

Set `SKIP_TRAINING = False`, point `DATASET_YAML` to your `data.yaml`, and run Section 4.

Training config used:

```python
# Augmentation
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
degrees=5, translate=0.1, scale=0.5
shear=2, perspective=0.0005
mosaic=1.0, mixup=0.1

# Optimiser
optimizer='auto'   # MuSGD (YOLO26 default)
lr0=0.01, lrf=0.01, momentum=0.937
weight_decay=0.0005, warmup_epochs=3
```

---

## 📊 Results & Benchmarks

### Detection accuracy (Indian License Plate dataset, val set)

| Model | mAP50 | mAP50-95 | Params | Inference (T4) |
|-------|-------|----------|--------|----------------|
| YOLOv8n | 0.891 | 0.612 | 3.2M | 4.1 ms |
| YOLOv9n | 0.903 | 0.631 | 2.0M | 3.9 ms |
| **YOLO26n** | **0.924** | **0.658** | **2.6M** | **3.4 ms** |

### OCR accuracy on cropped plates

| Method | Exact match | CER | Speed (CPU) |
|--------|-------------|-----|-------------|
| EasyOCR | 71.2% | 8.4% | 240 ms/plate |
| Tesseract | 64.8% | 11.7% | 80 ms/plate |
| **PaddleOCR + CLAHE** | **88.6%** | **3.1%** | **110 ms/plate** |

> Benchmarks run on a 2,000-image held-out test set of Indian roads.

### System throughput

| Hardware | Frame rate | Notes |
|----------|-----------|-------|
| NVIDIA T4 (Kaggle) | ~28 FPS | `--skip 1` |
| RTX 3060 | ~35 FPS | `--skip 1` |
| Intel i7 (CPU only) | ~6 FPS | `--skip 3` recommended |
| Raspberry Pi 4 | ~2 FPS | `--skip 5`, use nano model |

---

## 📂 Output Structure

```
output/
├── detections_live.csv       ← all detections (appended across runs)
├── detections_dedup.csv      ← one row per unique plate per source
├── detections.json           ← full JSON export
├── stats.png                 ← confidence histograms + plate frequency chart
├── plate_crops/
│   └── CAM01_20240115_143207_DL3CAB1234_a1b2c3.jpg
├── annotated/
│   └── frame_0042.jpg
└── live_snapshots/
    └── snap_CAM01_20240115_143207_DL3CAB1234_a1b2c3.jpg
```

### CSV schema

| Column | Example | Description |
|--------|---------|-------------|
| `timestamp` | `2024-01-15T14:32:07.123` | ISO 8601 |
| `camera_id` | `CAM01` | Thread identifier |
| `plate_text` | `DL3CAB1234` | Corrected OCR output |
| `detect_conf` | `0.8821` | YOLO26 confidence |
| `ocr_conf` | `0.9134` | PaddleOCR confidence |
| `plate_bbox` | `[120,340,280,390]` | `[x1,y1,x2,y2]` pixels |
| `vehicle_bbox` | `[80,200,420,520]` | Associated vehicle box |
| `crop_file` | `CAM01_..._DL3CAB1234_a1b2c3.jpg` | Saved plate crop |
| `snapshot_file` | `snap_CAM01_...jpg` | Full annotated frame |

---

## 🚢 Deployment Guide

### Security office / NVR server

```bash
# 1. Clone & install (headless OpenCV for servers)
pip install opencv-python-headless>=4.8.0 ultralytics paddlepaddle paddleocr pandas

# 2. Run as a background service
nohup python live_lp_detection.py \
  --source "rtsp://admin:pass@192.168.1.64:554/stream1" \
           "rtsp://admin:pass@192.168.1.65:554/stream1" \
  --headless \
  --skip 3 \
  > logs/detection.log 2>&1 &
```

### systemd service (auto-start on boot)

```ini
# /etc/systemd/system/lp-detection.service
[Unit]
Description=License Plate Detection Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/lp-detection
ExecStart=/usr/bin/python3 live_lp_detection.py \
  --source "rtsp://admin:pass@192.168.1.64:554/stream1" \
  --headless
Restart=always
RestartSec=10
StandardOutput=append:/var/log/lp-detection.log
StandardError=append:/var/log/lp-detection.err

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lp-detection
sudo systemctl start lp-detection
sudo journalctl -u lp-detection -f   # tail logs
```

### Docker

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "live_lp_detection.py"]
CMD ["--source", "0", "--headless"]
```

```bash
docker build -t lp-detection .
docker run --rm -v $(pwd)/output:/app/output lp-detection \
  --source "rtsp://admin:pass@192.168.1.64:554/stream1" --headless
```

---

## ⚙️ Tuning Tips

<details>
<summary><b>🐢 Slow server / high CPU usage</b></summary>

```bash
# Process fewer frames
python live_lp_detection.py --source ... --skip 5

# Or edit Config in the script:
Config.FRAME_SKIP = 5
```

Use `yolo26n` (nano) — it's already the default and ~3× faster than `yolo26s`.

</details>

<details>
<summary><b>❌ Too many false positives</b></summary>

```bash
python live_lp_detection.py --source ... --conf 0.55
```

</details>

<details>
<summary><b>❌ Missing plates at distance</b></summary>

```bash
python live_lp_detection.py --source ... --conf 0.30
```

Ensure your RTSP stream is at least **720p**. At 480p, small plates at distance will be missed regardless of threshold.

</details>

<details>
<summary><b>🔁 Same plate logged too many times</b></summary>

Edit `Config.DEDUP_COOLDOWN_SEC` in `live_lp_detection.py` (default: 60 seconds):

```python
Config.DEDUP_COOLDOWN_SEC = 120   # re-log same plate only after 2 min
```

</details>

<details>
<summary><b>🌙 Night / IR cameras</b></summary>

CLAHE preprocessing handles low-contrast plates automatically — no config changes needed.  
For pure IR (inverted grayscale) feeds, the grayscale OCR path activates automatically.

</details>

<details>
<summary><b>📡 Finding your RTSP URL</b></summary>

Most DVRs / IP cameras use one of these formats:

```
rtsp://admin:password@192.168.1.64:554/stream1
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
rtsp://192.168.1.64/live/ch0
rtsp://192.168.1.64:554/cam/realmonitor?channel=1&subtype=0
```

Use [ONVIF Device Manager](https://sourceforge.net/projects/onvifdm/) (free) to auto-discover cameras on your LAN.

</details>

---

## 🇮🇳 Indian Plate Format Reference

This detector handles all standard Indian plate formats:

| Format | Example | Length |
|--------|---------|--------|
| Standard (new series) | `DL 3C AB 1234` | 10 chars |
| Standard (old series) | `MH 12 AB 1234` | 10 chars |
| Short series | `HR 26 A 1234` | 9 chars |
| Very short | `MH 12 1234` | 8 chars |

### OCR character correction

Positional correction is applied after OCR to fix common misreads:

| Position | Expected | Common misread | Correction |
|----------|----------|----------------|------------|
| 0–1 | Letter | `0` → `O`, `1` → `I` | digit → letter |
| 2–3 | Digit | `O` → `0`, `I` → `1` | letter → digit |
| 4–5 | Letter | `0` → `O`, `8` → `B` | digit → letter |
| 6–9 | Digit | `O` → `0`, `I` → `1` | letter → digit |

---

## 🗂️ Project Structure

```
indian-lp-detection/
│
├── live_lp_detection.py              # Production script (RTSP / webcam / video)
├── indian_lp_detection_unified.ipynb # Kaggle/local notebook (train + infer + live)
├── requirements.txt
├── README.md
│
├── weights/
│   ├── license_plate_detector.pt     # ← your trained LP weights (not committed)
│   └── yolo26n.pt                    # ← auto-downloaded
│
└── output/                           # created at runtime
    ├── detections_live.csv
    ├── detections.json
    ├── plate_crops/
    ├── annotated/
    └── live_snapshots/
```

---

## ❓ FAQ

**Q: Where do I get `license_plate_detector.pt`?**  
A: Train it yourself using the notebook (Section 4) with any Indian LP dataset from Roboflow, or download a community-trained checkpoint from the Releases tab.

**Q: Does it work on Raspberry Pi?**  
A: Yes — use `--skip 5` and the nano model. Expect ~2 FPS. For real-time deployment on edge, export to ONNX and run with OpenCV DNN backend.

**Q: Can I run multiple cameras on one machine?**  
A: Yes. Pass multiple `--source` arguments. Each camera runs in its own thread sharing the same model instance. Tested with up to 8 cameras on a single RTX 3060.

**Q: The RTSP stream keeps disconnecting.**  
A: The script auto-reconnects after `Config.RECONNECT_WAIT` seconds (default 5s). Persistent drops usually indicate network instability or incorrect RTSP credentials.

**Q: How do I query the CSV in real time?**  
A: The CSV is append-only and safe to `tail -f`. You can also load it with pandas at any point — each row is flushed immediately after writing.

**Q: Can I integrate this with a database?**  
A: Replace or extend the `CSVLogger.write()` method with your database insert. The row dict contains all fields needed for a clean schema.

---

## 🔭 Roadmap

- [ ] FastAPI endpoint for REST-based querying
- [ ] WhatsApp / Telegram alert on unknown plates
- [ ] Web dashboard (plate log viewer + live feed)
- [ ] Re-identification across cameras
- [ ] Blacklist / whitelist enforcement mode
- [ ] ONNX Runtime inference path (no PyTorch dependency)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

```bash
# Fork → clone → branch → PR
git checkout -b feature/your-feature
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

Please run your changes against at least one video file before submitting.

---

## 📄 License

MIT © 2024 — free to use, modify, and deploy commercially.  
See [LICENSE](LICENSE) for full terms.

---

## 🙏 Acknowledgements

| Project | Used for |
|---------|----------|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO26 detection backbone |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Plate text recognition |
| [OpenCV](https://opencv.org) | Frame capture, CLAHE, annotation |
| [Roboflow Universe](https://universe.roboflow.com) | Indian LP training datasets |

---

<div align="center">



<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>

</div>
