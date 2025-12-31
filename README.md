# LipSyncing videos with LatentSync

This is a web-based lip-synchronization service using FastAPI, deployed on Modal. This service accepts a video file and a separate audio file as inputs and generates a new video that is lipsynced to the audio file.

We do this by using the **LatentSync** model but we also support Wav2Lip.

We also provide a script to benchmark the runtimes of both LatentSync and Wav2Lip on different GPUs. 

---

## Why LatentSync?

- Sharper image of the lip area - diffusion based model
- Better syncing
- Near real-time video generation (10 mins for a 1 min video)

---

## 1. Environment & Dependency Setup

### 1.1 Create a Conda Environment
We recommend using Python **3.10**.

```bash
conda create -n lipsync python=3.10 -y
conda activate lipsync
```

### 1.2 Install Modal

```bash
pip install modal
python3 -m modal setup
```

---

## 2. Clone Required Repositories

clone the **modified versions** of Wav2Lip and LatentSync.

```bash
git clone https://github.com/shrutijain1405/LatentSync.git
git clone https://github.com/shrutijain1405/Wav2Lip.git
```
---

## 3. Download Model Checkpoints

Follow the README instructions of the above Repositories and download the correct model checkpoints.

---

## 4. Deploy the Modal App

```bash
modal deploy lipsyncApp.py
```

---

## 5. Use the Web UI

Open `index.html` in your browser to upload your video and audio files, submit the job and download the result.

---

## 6. Benchmarking

```bash
python benchmarking_gpus.py
```

Results are stored in `benchmark_pipelines.csv`.

---

## 7. Qualitative Results

<!-- | Wav2Lip | LatentSync |
|--------|------------|
| ![](outputs/output_video_yt1_wav2lip.mp4) | ![](outputs/output_video_yt1_latentSync.mp4) |
| ![](outputs/output_video_yt2_wav2lip.mp4) | ![](outputs/output_video_yt2_latentSync.mp4) | -->


### Sample Video 1 — Wav2Lip
[![Wav2Lip Video 1](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_wav2lip_thumbnail.png)](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_wav2lip.mp4)

### Sample Video 1 — LatentSync
[![LatentSync Video 1](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_latentSync_thumbnail.png)](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_latentSync.mp4)

### Sample Video 2 — Wav2Lip
[![Wav2Lip Video 2](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt2_wav2lip_thumbnail.png)](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt2_wav2lip.mp4)

### Sample Video 2 — LatentSync
[![LatentSync Video 2](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt2_latentSync_thumbnail.png)](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt2_latentSync.mp4)


---

## 8. MuseTalk (explored, but not used)

MuseTalk was explored but discarded due to poor preservation of facial attributes (e.g. moustache).

[![MuseTalk Video 1](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_museTalk_thumbnail.png)](https://raw.githubusercontent.com/shrutijain1405/lipsync/main/outputs/output_video_yt1_museTalk.mp4)

---

## 9. Benchmark Results

| Pipeline     | GPU  | Sample 1 (sec) | Sample 2 (sec) |
|-------------|------|---------------:|---------------:|
| Wav2Lip     | A100 | 139.76 | 112.19 |
| Wav2Lip     | L4   | 204.77 | 163.88 |
| Wav2Lip     | H100 | **110.22** | **91.01** |
| LatentSync  | A100 | 623.28 | 522.95 |
| LatentSync  | L4   | 2080.27 | 1770.15 |
| LatentSync  | H100 | **405.04** | **338.58** |

---

## 10. Limitations

- Single-face only
- Sensitive to occlusion
- High VRAM usage

---