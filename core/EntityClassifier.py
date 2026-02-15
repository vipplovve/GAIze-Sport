import cv2
import numpy as np

class EntityClassifier:
    def __init__(self):
        self.classes = ["Team A", "Team B", "Referee"]

    def classify_object(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return "Unknown"
        
        avg_color_per_row = np.average(image_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        if avg_color[0] > avg_color[2]:
            return "Team A"
        else:
            return "Team B"

    def predict_batch(self, crops):
        return [self.classify_object(crop) for crop in crops]
