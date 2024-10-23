# WROOM Technical Assessment 1, ComfyUI wrapper

import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionPipeline, ControlNetModel

class ComfyUIWrapper:
    def __init__(self, sd_model_path, controlnet_path):
        self.device = "cpu"
        
        # Load the ControlNet model from the ControlNet directory
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path).to(self.device)
        
        # Load the Stable Diffusion model from the Stable Diffusion directory
        self.pipeline = StableDiffusionPipeline.from_pretrained(sd_model_path, controlnet=self.controlnet).to(self.device)

    def preprocess_frames(self, video_path):
        # Extract frames from video and preprocess them
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (512, 512))  # Adjust frame size
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            frame_count += 1
        cap.release()
        print(f"Total frames extracted from video: {frame_count}")
        return frames

    def generate_frame(self, input_frame):
        #Generate a frame with ControlNet and Stable Diffusion
        # Convert frame to torch tensor
        input_tensor = torch.tensor(input_frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.pipeline(input_tensor)
        
        # Post-process the generated image
        output_image = output.images[0].cpu().numpy()
        return (output_image * 255).astype(np.uint8)

    def generate_video(self, input_video_path, output_video_path):
        #Generate video from input video using ControlNet and Stable Diffusion
        frames = self.preprocess_frames(input_video_path)
        generated_frames = []

        for idx, frame in enumerate(frames):
            print(f"Processing frame {idx+1}/{len(frames)}")
            generated_frame = self.generate_frame(frame)
            generated_frames.append(generated_frame)
        
        print(f"Total generated frames: {len(generated_frames)}")
        
        if generated_frames:
            self.save_video(generated_frames, output_video_path)
        else:
            print("No frames were generated, output video will not be saved.")

    def save_video(self, frames, output_path):
        #Save list of frames to video
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()
        print(f"Video saved at {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, required=True, help="Path to input video")
    parser.add_argument('--output_video_path', type=str, required=True, help="Path to output video")
    
    args = parser.parse_args()

    # Stable Diffusion model weights are in "models/StableDiffusion/"
    # ControlNet model weights are in "models/ControlNet/"
    wrapper = ComfyUIWrapper("models/StableDiffusion/", "models/ControlNet/")
    wrapper.generate_video(args.input_video_path, args.output_video_path)
