import cv2
import numpy as np

class MiniMapEngine:
    def __init__(self, pitch_image_path="assets/pitch_template.png"):
        self.pitch_image_path = pitch_image_path
        self.pitch_img = None
        self.homography_matrix = None
        self.src_pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.dst_pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])

    def load_pitch(self):
        try:
            self.pitch_img = cv2.imread(self.pitch_image_path)
            if self.pitch_img is None:
                raise FileNotFoundError
        except:
            print("Pitch template not found. Creating a blank one.")
            self.pitch_img = np.zeros((600, 400, 3), dtype=np.uint8)
            cv2.rectangle(self.pitch_img, (20, 20), (380, 580), (0, 255, 0), 2)

    def compute_homography(self, src_pts):
        self.src_pts = np.array(src_pts, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(self.src_pts, self.dst_pts)

    def transform_point(self, point):
        if self.homography_matrix is None:
            return point
        
        pts = np.float32([[point]]).reshape(-1, 1, 2)
        dst_pts = cv2.perspectiveTransform(pts, self.homography_matrix)
        return (int(dst_pts[0][0][0]), int(dst_pts[0][0][1]))

    def draw_minimap(self, player_positions):
        if self.pitch_img is None:
            self.load_pitch()

        minimap = self.pitch_img.copy()
        
        for pos in player_positions:
            map_pos = (int(pos[0] / 1920 * 400), int(pos[1] / 1080 * 600))
            cv2.circle(minimap, map_pos, 5, (0, 0, 255), -1)

        return minimap
