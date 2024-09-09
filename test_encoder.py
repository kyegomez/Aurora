import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import cv2
import numpy as np

# Load the model
model = (
    AutoModel.from_pretrained(
        "OpenGVLab/InternViT-300M-448px",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    .cuda()
    .eval()
)

# Load the image processor
image_processor = CLIPImageProcessor.from_pretrained(
    "OpenGVLab/InternViT-300M-448px"
)


# Function to sample frames from the video
def sample_frames(video_path, duration):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    if duration < 4:
        num_frames = 4
    elif duration < 16:
        num_frames = int(duration)
    else:
        num_frames = 16

    frame_interval = max(int(fps), 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = frame_count / fps
    frame_indices = np.linspace(
        0, frame_count - 1, num_frames, dtype=int
    )

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# Function to convert video frames to tensors
def process_video_frames(frames):
    tensors = []
    for frame in frames:
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ).convert("RGB")
        pixel_values = image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        tensors.append(pixel_values)

    # Stack all frame tensors into a single tensor
    return torch.stack(tensors).squeeze(1).to(torch.bfloat16).cuda()


# Video processing pipeline
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    # Sample frames
    frames = sample_frames(video_path, duration)

    # Convert sampled frames to tensors
    video_tensor = process_video_frames(frames)

    # Feed the video tensor into the model
    with torch.no_grad():
        outputs = model(video_tensor)

    return outputs


# Example usage
video_path = "swarms_workshop_promo.mp4"
outputs = process_video(video_path)
print(outputs)
