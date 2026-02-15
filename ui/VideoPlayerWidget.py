import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import tkinter as tk

class VideoPlayerWidget(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.controls_frame = ctk.CTkFrame(self, height=40)
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)

        self.btn_play = ctk.CTkButton(self.controls_frame, text="▶ Play", width=70, command=self.toggle_play)
        self.btn_play.pack(side="left", padx=3)

        self.btn_stop = ctk.CTkButton(self.controls_frame, text="⬛ Stop", width=70, command=self.stop_video, fg_color="#8B0000", hover_color="#B22222")
        self.btn_stop.pack(side="left", padx=3)

        self.btn_prev = ctk.CTkButton(self.controls_frame, text="◀", width=40, command=self.prev_frame)
        self.btn_prev.pack(side="left", padx=3)

        self.btn_next = ctk.CTkButton(self.controls_frame, text="▶", width=40, command=self.next_frame)
        self.btn_next.pack(side="left", padx=3)

        self.slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, command=self.seek)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)

        self.lbl_frame_info = ctk.CTkLabel(self.controls_frame, text="0 / 0", width=80, font=("Roboto", 11))
        self.lbl_frame_info.pack(side="right", padx=5)

        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.video_path = None
        self.photo = None
        self.video_width = 640
        self.video_height = 360
        self.frame_processor = None

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas_w = 640
        self._canvas_h = 360

    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._canvas_h = event.height

    def load_video(self, path):
        self.video_path = path
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")

        self.update_frame()

    def toggle_play(self):
        if not self.video_path:
            return

        self.is_playing = not self.is_playing
        self.btn_play.configure(text="⏸ Pause" if self.is_playing else "▶ Play")

        if self.is_playing:
            self.play_video()

    def stop_video(self):
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        if self.cap:
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_frame()

    def _process(self, frame):
        if self.frame_processor:
            return self.frame_processor(frame)
        return frame

    def play_video(self):
        if self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.set((self.current_frame / self.total_frames) * 100)
                self.lbl_frame_info.configure(text=f"{self.current_frame} / {self.total_frames}")
                self.display_frame(self._process(frame))
                delay = max(1, int(1000 / self.fps))
                self.after(delay, self.play_video)
            else:
                self.is_playing = False
                self.btn_play.configure(text="▶ Play")

    def prev_frame(self):
        if self.cap and self.current_frame > 0:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play")
            self.current_frame = max(0, self.current_frame - 2)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()

    def next_frame(self):
        if self.cap and self.current_frame < self.total_frames:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play")
            self.update_frame()

    def seek(self, value):
        if self.cap:
            target_frame = int((float(value) / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame
            self.update_frame()

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.lbl_frame_info.configure(text=f"{self.current_frame} / {self.total_frames}")
                self.slider.set((self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0)
                self.display_frame(self._process(frame))

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        cw = self._canvas_w if self._canvas_w > 1 else 640
        ch = self._canvas_h if self._canvas_h > 1 else 360

        w, h = img.size
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")
