from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from audio_utils import mp3_to_mel
from model import load_model, GENRE_MAP, DEVICE

app = FastAPI()

model = load_model("resnet_genre_best.pth", num_classes=22)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    mel_segments = mp3_to_mel(audio_bytes)
    probs_list = []

    with torch.no_grad():
        for mel in mel_segments:
            mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            logits = model(mel)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())
        
    probs_all = torch.cat(probs_list, dim=0)
    probs_mean = probs_all.mean(dim=0, keepdim=True)

    topk = torch.topk(probs_mean, k=1)

    return {
        "num_segments": len(mel_segments),
        "topk": [
            {"genre": GENRE_MAP[i.item()], "prob": float(p)}
            for i, p in zip(topk.indices[0], topk.values[0])
        ]
    }

@app.get("/")
def index():
    return FileResponse('index.html')