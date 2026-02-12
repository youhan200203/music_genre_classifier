import io
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 48
N_LENGTHS = 512
SEGMENT_SEC = 20
STRIDE_SEC = 10


def mp3_to_mel(file_bytes, n_mels=48, target_len=1024):
    mp3 = io.BytesIO(file_bytes)
    y, sr = librosa.load(mp3, sr=SAMPLE_RATE, mono=True)
    segment_samples = SEGMENT_SEC * SAMPLE_RATE
    stride_samples = STRIDE_SEC * SAMPLE_RATE
    mel_segments = []
    for start in range(0, len(y) - segment_samples + 1, stride_samples):
        y_seg = y[start:start + segment_samples]
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=n_mels, hop_length=512).astype(np.float32)
        mel = librosa.power_to_db(mel, ref=np.max)
        t = mel.shape[1]
        if t > target_len:
            mel = mel[:, :target_len]
        elif t < target_len:
            mel = np.pad(mel, ((0,0),(0,target_len - t)), mode='constant')
        mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-6)
        mel_segments.append(mel)
    return mel_segments
