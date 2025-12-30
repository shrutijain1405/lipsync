import modal
import csv
from pathlib import Path

APP_NAME = "lipsync-modalApp"
INPUTS = [
    ("inputs/input_video_yt1.mp4", "inputs/input_audio_yt1.wav"),
    ("inputs/input_video_yt2.mp4", "inputs/input_audio_yt2.wav"),
]
GPUS = ["A100", "L4", "H100"]
PIPELINES = ["latentSync", "wav2lip"]

results = []

for pipeline in PIPELINES:
    for gpu_name in GPUS:
        for idx, (video, audio) in enumerate(INPUTS):
            output_path = f"outputs/benchmark_{pipeline}_{gpu_name}_out_{idx+1}.mp4"

            fn = modal.Function.from_name(APP_NAME, f"benchmark_{gpu_name}")

            print(f"Running {pipeline} on {gpu_name} sample {idx+1}...")
            print(fn)
            inf_time = fn.remote(video, audio, output_path, pipeline)

            results.append({
                "pipeline": pipeline,
                "gpu": gpu_name,
                "sample": idx+1,
                "video": video,
                "audio": audio,
                "time_sec": inf_time
            })

            print(f"{pipeline} | {gpu_name} sample {idx+1}: {inf_time:.2f}s")

Path("benchmark_pipelines.csv").parent.mkdir(exist_ok=True)

with open("benchmark_pipelines.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["pipeline", "gpu", "sample", "video", "audio", "time_sec"]
    )
    writer.writeheader()
    writer.writerows(results)

print("done.")
