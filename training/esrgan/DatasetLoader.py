import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DatasetLoader(Dataset):
    def __init__(self, hr_dir, hr_crop_size=128, scale_factor=4, augment=True):
        super(DatasetLoader, self).__init__()
        
        self.hr_dir = hr_dir
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // scale_factor
        self.scale_factor = scale_factor
        self.augment = augment
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        self.image_paths = [
            os.path.join(hr_dir, f) for f in sorted(os.listdir(hr_dir))
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {hr_dir}")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert('RGB')
        
        w, h = hr_img.size
        if w < self.hr_crop_size or h < self.hr_crop_size:
            hr_img = hr_img.resize(
                (max(w, self.hr_crop_size), max(h, self.hr_crop_size)),
                Image.BICUBIC
            )
        
        i, j, th, tw = transforms.RandomCrop.get_params(
            hr_img, output_size=(self.hr_crop_size, self.hr_crop_size)
        )
        hr_img = TF.crop(hr_img, i, j, th, tw)
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = TF.hflip(hr_img)
            if random.random() > 0.5:
                hr_img = TF.vflip(hr_img)
            if random.random() > 0.5:
                hr_img = TF.rotate(hr_img, 90)
        
        lr_img = hr_img.resize(
            (self.lr_crop_size, self.lr_crop_size),
            Image.BICUBIC
        )
        
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        
        return lr_tensor, hr_tensor

class VideoFrameDataset(DatasetLoader):
    @staticmethod
    def extract_frames(video_path, output_dir, every_n_frames=5):
        import cv2
        
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % every_n_frames == 0:
                filename = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(filename, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count

if __name__ == "__main__":
    test_dir = "test_hr_images"
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(os.path.join(test_dir, f"test_{i}.png"))
    
    dataset = DatasetLoader(test_dir, hr_crop_size=128, scale_factor=4)
    lr, hr = dataset[0]
    
    import shutil
    shutil.rmtree(test_dir)
