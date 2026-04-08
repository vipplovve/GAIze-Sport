import cv2
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training' / 'esrgan'))

class FrameEnhancementEngine:
    def __init__(self, model_path="training/esrgan/checkpoints/esrgan_generator.pth", num_blocks=8):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        from ESRGANModel import RRDBNet

        print(f"Loading ESRGAN model from {self.model_path}...")
        try:
            self.model = RRDBNet(num_blocks=self.num_blocks).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"ESRGAN model loaded on {self.device}.")
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            self.model = None
        except Exception as e:
            print(f"Error loading ESRGAN model: {e}")
            self.model = None

    def enhance_frame(self, frame):
        if self.model is None:
            self.load_model()

        if self.model is None:
            return frame

        orig_h, orig_w = frame.shape[:2]
        max_dim = 480

        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            process_frame = frame

        img_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr_tensor = self.model(tensor)

        sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        target_w = orig_w * 4
        target_h = orig_h * 4

        final_bgr = cv2.resize(sr_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return final_bgr

    def enhance_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 4, h * 4))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            enhanced = self.enhance_frame(frame)
            writer.write(enhanced)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Enhanced {frame_count}/{total} frames...")

        cap.release()
        writer.release()
        print(f"Enhanced video saved to {output_path}")
        return output_path
