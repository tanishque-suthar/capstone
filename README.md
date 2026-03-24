# Capstone: Video Surveillance & Analysis Platform

A high-performance video surveillance as-a-service (VSaaS) platform that integrates AI-powered perception (YOLOv11), entity tracking (BoT-SORT), and semantic search (RAG using SigLIP and LanceDB) for real-time video analysis and forensic search.

## 🚀 Features

- **Automated Perception Pipeline**: Trigger-based event detection using MOG2 background subtraction and entropy analysis.
- **Entity Detection & Tracking**: Real-time object detection (cars, persons, etc.) and trajectory tracking using YOLOv11 and BoT-SORT.
- **RAG-Powered Forensic Search**: Semantic search over entity crops using SigLIP embeddings and LanceDB vector storage.
- **Interactive Video Annotator**: Web-based UI for visualizing detected tracks, entity crops, and event timelines.
- **Automated Reporting**: Generation of event summaries and handoffs for downstream processing.

## 🛠️ Project Structure

- `/app`: FastAPI backend handling ingestion, perception, and RAG pipelines.
- `/frontend`: Next.js 15+ application with a modern dashboard UI.
- `/config`: Configuration files for homography and pipeline settings.
- `/dataset`: Local storage for video clips, entity crops, and the LanceDB index.

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 20+
- `ffmpeg` (for video processing)

### 1. Backend Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database (if not already present):
   The application uses SQLite (`event_registry.db`) and LanceDB (in `dataset/lancedb`).

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install packages:
   ```bash
   npm install
   ```

### 3. Running the Application
You need to run both the backend and frontend servers simultaneously.

**Backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev -- -p 3000
```

Access the dashboard at `http://localhost:3000`.

## 🛠️ Common Issues & Troubleshooting

- **C++ Build Tools (Windows)**: If `pip install` fails on `lapx`, `sentencepiece`, or other packages, you likely need the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Select the "Desktop development with C++" workload.
- **FFMPEG**: The backend requires `ffmpeg` to be in your system PATH for video archival. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
- **Video Paths**: When triggering the pipeline via the UI or API, you must provide the **absolute path** to the video file on your local machine (e.g., `C:/videos/my_video.mp4`).
- **Models**: On first run, the app will download YOLO weights (`yolo11n.pt`) and SigLIP weights (~1GB). Ensure you have a stable internet connection.

## 🧪 Testing & Debugging
- Backend API Docs: `http://localhost:8000/docs`
- RAG Pipeline Test: `python test_rag.py`
- SigLIP Debugging: `python debug_siglip.py`
