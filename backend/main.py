from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import shutil, os, time, threading

from model import process_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER  = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

# ── Global status dict (read by /status) ──
processing_status = {
    "status":          "idle",
    "progress":        0,
    "result":          None,
    "processing_time": 0,
    "error_message":   None,
}


def run_processing(input_path, output_path):
    global processing_status
    start = time.time()
    try:
        processing_status["status"]   = "processing"
        processing_status["progress"] = 5

        json_data = process_video(input_path, output_path)

        # Model returned an error dict
        if isinstance(json_data, dict) and "error" in json_data:
            raise Exception(json_data["error"])

        if not os.path.exists(output_path):
            raise Exception("Output video was not created.")

        deliveries = json_data.get("deliveries", [])
        if len(deliveries) == 0:
            print("⚠️  Warning: 0 deliveries detected — ball may not have been visible.")

        processing_status.update({
            "status":          "completed",
            "progress":        100,
            "processing_time": round(time.time() - start, 2),
            "error_message":   None,
            "result": {
                "video_url": "http://localhost:8000/outputs/processed_latest.mp4",
                "json_data": json_data,
            },
        })
        print(f"✅ Done in {processing_status['processing_time']}s — {len(deliveries)} deliveries")

    except Exception as e:
        print("❌ Processing error:", e)
        processing_status.update({
            "status":        "error",
            "error_message": str(e),
            "result":        {"error": str(e)},
        })


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global processing_status

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed = [".mp4", ".avi", ".mov", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed}")

    input_path  = os.path.join(UPLOAD_FOLDER, "latest.mp4")
    output_path = os.path.join(OUTPUT_FOLDER, "processed_latest.mp4")

    processing_status = {
        "status": "processing", "progress": 0,
        "result": None, "processing_time": 0, "error_message": None,
    }

    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    print(f"📥 Received: {file.filename}")
    threading.Thread(target=run_processing, args=(input_path, output_path), daemon=True).start()
    return {"message": "Processing started"}


@app.get("/status")
def get_status():
    return processing_status
