from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse
from diffusers import StableDiffusionPipeline
import torch
import threading
import time
from collections import OrderedDict
import os
from PIL import Image
import imageio

app = FastAPI()

# =========================
# ⚡ CPU OPTIMIZATION
# =========================
torch.set_num_threads(2)

# =========================
# 🔥 MODEL
# =========================
model_name = "Lykon/dreamshaper-7"

pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    safety_checker=None
)

pipe = pipe.to("cpu")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# =========================
# 📦 STORAGE
# =========================
db = OrderedDict()
queue = []
progress_db = {}

MAX_HISTORY = 40
NUM_WORKERS = 1

IMG_DIR = "image"
os.makedirs(IMG_DIR, exist_ok=True)

# =========================
# ✂️ SPLIT INTO 12
# =========================
def split_image_into_12(img_path: str):
    img = Image.open(img_path)

    w, h = img.size
    cols, rows = 4, 3

    tile_w = w // cols
    tile_h = h // rows

    base = img_path.replace(".png", "")

    index = 1

    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = left + tile_w
            bottom = top + tile_h

            crop = img.crop((left, top, right, bottom))

            out_path = f"{base}_{index}.png"
            crop.save(out_path)

            index += 1


# =========================
# 🎬 SPLIT VIDEO FRAMES
# =========================
def split_video_frames(frames, base_path):
    for idx, frame in enumerate(frames):
        frame_path = f"{base_path}_frame_{idx}.png"
        frame.save(frame_path)

        split_image_into_12(frame_path)


# =========================
# 🚀 GENERATION ENGINE
# =========================
def generate_ai_stream(message: str, mode="fast"):

    try:
        start = time.time()

        # ⚡ режимы
        if mode == "fast":
            steps = 2
            cfg = 1.5
            num_frames = 4
        else:
            steps = 6
            cfg = 3.0
            num_frames = 8

        progress_db[message] = 0

        # 🔥 базовый кадр
        base_image = pipe(
            message,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=192,
            width=192
        ).images[0]

        frames = [base_image]

        # 🔥 генерация видео (итеративно)
        for i in range(1, num_frames):

            progress_db[message] = int((i / num_frames) * 100)

            frame = pipe(
                message,
                num_inference_steps=2,  # ⚡ дешево
                guidance_scale=1.2,
                height=192,
                width=192
            ).images[0]

            frames.append(frame)

        # =========================
        # 🎬 SAVE VIDEO
        # =========================
        filename = f"{IMG_DIR}/video_{int(time.time()*1000)}.mp4"

        imageio.mimsave(
            filename,
            frames,
            fps=4
        )

        # =========================
        # ✂️ SPLIT EACH FRAME
        # =========================
        split_video_frames(frames, filename.replace(".mp4", ""))

        progress_db[message] = 100

        duration = round(time.time() - start, 2)

        result = f"{filename} | {mode} | {duration}s"

    except Exception as e:
        result = f"error: {str(e)}"

    if message in db:
        db[message]["reply"] = result
        db[message]["status"] = "done"

    progress_db.pop(message, None)

    return result


# =========================
# 🔄 WORKER
# =========================
def worker():
    while True:
        if queue:
            message, mode = queue.pop(0)

            if message in db and db[message]["status"] == "done":
                continue

            generate_ai_stream(message, mode)
        else:
            time.sleep(0.03)


threading.Thread(target=worker, daemon=True).start()


# =========================
# 🌐 API
# =========================

@app.get("/")
async def root():
    return PlainTextResponse("🎬 AI Video Generator Running")


@app.get("/fast")
async def fast(message: str):

    if message not in db:
        db[message] = {"status": "pending", "reply": ""}
        queue.append((message, "fast"))

        if len(db) > MAX_HISTORY:
            db.popitem(last=False)

    return PlainTextResponse("accepted")


@app.get("/quality")
async def quality(message: str):

    if message not in db:
        db[message] = {"status": "pending", "reply": ""}
        queue.append((message, "quality"))

        if len(db) > MAX_HISTORY:
            db.popitem(last=False)

    return PlainTextResponse("accepted")


@app.get("/get")
async def get(message: str):

    if message not in db:
        return PlainTextResponse("not found")

    data = db[message]

    if data["status"] == "pending":
        progress = progress_db.get(message, 0)
        return PlainTextResponse(f"generating video... {progress}%")

    return PlainTextResponse(data["reply"])


@app.get("/video")
async def get_video(path: str):

    if not os.path.exists(path):
        return PlainTextResponse("file not found")

    return FileResponse(path)


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    import os

port = int(os.environ.get("PORT", 10000))

uvicorn.run(app, host="0.0.0.0", port=port)
