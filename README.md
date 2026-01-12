# LipSyncing videos with LatentSync

This is a web-based lip-synchronization service using FastAPI, deployed on Modal. This service accepts a video file and a separate audio file as inputs and generates a new video that is lipsynced to the audio file.

We do this by using the **LatentSync** model but we also support Wav2Lip.

We also provide a script to benchmark the runtimes of both LatentSync and Wav2Lip on different GPUs. 

You can find my slides here: https://docs.google.com/presentation/d/19HnPf5GJuIK7sEFEZvO3fmb3Vb794zFoW9JlxAxQfCM/edit?usp=sharing

---

## Why LatentSync?

- Sharper image of the lip area - diffusion based model
- Better syncing
- Near real-time video generation (8 mins for a 1 min video)

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

## 7. Qualitative Results of LatentSync

https://github.com/user-attachments/assets/1b4dba80-54cb-4fb4-b271-79a3ce570fe8

https://github.com/user-attachments/assets/f3353001-20a4-41d1-aa67-5c572f426c11

## 8. Other models

### Wav2Lip

Wav2Lip blurs the region near the lip, the blending of the lip region is not smooth (in some frames bounding box is visible) and unwanted artifacts can be seen

https://github.com/user-attachments/assets/6668d3e5-3895-4187-8594-341851e9efd6

https://github.com/user-attachments/assets/20a1b01d-525b-4ee9-8b60-1388e9f2ad3c


### MuseTalk

MuseTalk has poor preservation of facial attributes (e.g. moustache).

https://github.com/user-attachments/assets/e8656aff-7e0d-48d9-b56b-3b3691301e3b

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
