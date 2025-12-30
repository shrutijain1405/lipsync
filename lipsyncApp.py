# lipsyncApp.py
import modal
import os, time, shutil
import subprocess

app = modal.App("lipsync-modalApp")

volume = modal.Volume.from_name("lipsync-data", create_if_missing=True)

image = ( #latentsync and wav2lip
    modal.Image.debian_slim()
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1" #latentSync
    )
    .pip_install(
        # # Wav2Lip dependencies
        # "librosa",
        # "numpy",
        # "opencv-python",
        "opencv-contrib-python",
        # "torch",
        # "torchvision",
        "tqdm",
        "numba",

        # LatentSync dependencies
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "decord",
        "accelerate",
        "einops",
        "omegaconf",
        "opencv-python",
        "mediapipe",
        "python_speech_features",
        "librosa",
        "scenedetect",
        "ffmpeg-python",
        "imageio",
        "imageio-ffmpeg",
        "lpips",
        "face-alignment",
        "numpy",
        "kornia",
        "insightface",
        "onnxruntime-gpu",
        "DeepCache",

        # Your app deps
        "fastapi[standard]"
    )
    .add_local_dir(
        ".",  
        remote_path="/root"
    )
)


def getLipSyncedVideoWav2Lip(videoPath, audioPath, outputPath, jobId):

    os.makedirs(f"/data/{jobId}/temp", exist_ok=True)
    
    subprocess.run(
        [
            "python",
            "/root/Wav2Lip/inference.py",
            "--checkpoint_path", "/root/Wav2Lip/checkpoints/wav2lip_gan.pth",
            "--face", videoPath,
            "--audio", audioPath,
            "--outfile", outputPath,
            "--job_id", jobId
            # "--face_det_batch_size", str(8) 
        ],
        check=True
    )

    shutil.rmtree(f"/data/{jobId}/temp")

def getLipSyncedVideoLatentSync(videoPath, audioPath, outputPath):

    subprocess.run(
    [
            "python",
            "-m", "LatentSync.scripts.inference",
            "--unet_config_path", "/root/LatentSync/configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "/root/LatentSync/checkpoints/latentsync_unet.pt",
            "--inference_steps", "20",
            "--guidance_scale", "1.5",
            "--enable_deepcache",
            "--video_path", videoPath,
            "--audio_path", audioPath,
            "--video_out_path", outputPath
        ],
        check=True
    )


def runBenchmark(video_path: str, audio_path: str, output_path: str, pipeline: str):
    start = time.perf_counter()

    if(pipeline == "latentSync"):
        getLipSyncedVideoLatentSync(video_path, audio_path, output_path)
    elif(pipeline == "wav2lip"):
        getLipSyncedVideoWav2Lip(video_path, audio_path, output_path,"0000000")
    else:
        print("ERROR! not a valid pipeline")
    end = time.perf_counter()

    return end - start

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=4000,
    gpu="A100-40GB"
)
def benchmark_A100(video, audio, output_path, pipeline):
    return runBenchmark(video, audio, output_path, pipeline)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=4000,
    gpu="H100!"
)
def benchmark_H100(video, audio, output_path, pipeline):
    return runBenchmark(video, audio, output_path, pipeline)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=6000,
    gpu="L4"
)
def benchmark_L4(video, audio, output_path, pipeline):
    return runBenchmark(video, audio, output_path, pipeline)


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",      
    timeout=2000
)
def runLipsync(jobId: str, videoPath: str, audioPath: str):
    from jobStore import JobStore
    from jobStatus import JobStatus
    jobStore = JobStore()

    try:
        jobStore.setJobStatus(jobId, JobStatus.RUNNING)

        if not os.path.exists(videoPath):
            raise FileNotFoundError(f"Video not found: {videoPath}")

        if not os.path.exists(audioPath):
            raise FileNotFoundError(f"Audio not found: {audioPath}")
        
        outputPath = outputPath = f"/data/{jobId}/output_video.mp4"
        
        # getLipSyncedVideoLatentSync(videoPath, audioPath, outputPath)
        getLipSyncedVideoWav2Lip(videoPath,audioPath,outputPath,jobId)
        
        jobStore.setJobStatus(jobId, JobStatus.COMPLETED)

    except Exception as e:
        jobStore.setJobStatus(jobId, JobStatus.FAILED)
        raise e


@app.function(image=image, volumes={"/data": volume})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():

    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from jobStore import JobStore

    webApp = FastAPI()
    webApp.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    jobStore = JobStore()

    @webApp.post("/submit")
    async def submitJob(
        video: UploadFile = File(...),
        audio: UploadFile = File(...)
    ):
        import uuid

        jobId = str(uuid.uuid4())
        jobStore.createJob(jobId)

        job_dir = f"/data/{jobId}"
        os.makedirs(job_dir, exist_ok=True)

        video_path = f"{job_dir}/input_video.mp4"
        audio_path = f"{job_dir}/input_audio.wav"

        with open(video_path, "wb") as vf:
            shutil.copyfileobj(video.file, vf)

        with open(audio_path, "wb") as af:
            shutil.copyfileobj(audio.file, af)

        volume.commit()
        
        runLipsync.spawn(jobId, video_path, audio_path)

        return {"jobId": jobId}

    @webApp.get("/status/{jobId}")
    def getStatus(jobId: str):
        return { "jobId": jobId,
                "status": jobStore.getJobStatus(jobId)
                }
    

    @webApp.get("/result/{jobId}")
    def getResult(jobId: str):
        if jobStore.isJobReady(jobId) == False:
            return {"Error": "Result not ready"}

        return FileResponse(
            f"/data/{jobId}/output_video.mp4",
            media_type="video/mp4",
            filename=f"{jobId}.mp4"
        )

    return webApp
