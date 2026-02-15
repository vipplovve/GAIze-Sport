import customtkinter as ctk
import tkinter as tk
from threading import Thread
import time

class CalibrationView(ctk.CTkToplevel):
    def __init__(self, parent, frame, callback):
        super().__init__(parent)
        self.title("Calibrate Minimap")
        self.geometry("800x600")
        
        self.callback = callback
        self.points = []
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = frame_rgb.shape
        self.pil_image = Image.fromarray(frame_rgb)
        
        aspect = self.img_w / self.img_h
        display_w = 780
        display_h = int(display_w / aspect)
        self.resized_image = self.pil_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.resized_image)
        
        self.scale_x = self.img_w / display_w
        self.scale_y = self.img_h / display_h
        
        self.lbl_instruct = ctk.CTkLabel(self, text="Click 4 corners of the pitch (TL, TR, BR, BL)", font=("Roboto", 12))
        self.lbl_instruct.pack(pady=5)
        
        self.canvas = ctk.CTkCanvas(self, width=display_w, height=display_h, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.btn_done = ctk.CTkButton(self, text="Done", state="disabled", command=self.finish)
        self.btn_done.pack(pady=10)

    def on_click(self, event):
        if len(self.points) >= 4:
            return
            
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white")
        self.canvas.create_text(x, y-15, text=str(len(self.points)+1), fill="yellow")
        
        real_x = int(x * self.scale_x)
        real_y = int(y * self.scale_y)
        self.points.append([real_x, real_y])
        
        if len(self.points) == 4:
            self.btn_done.configure(state="normal")
            self.lbl_instruct.configure(text="Points selected. Click Done.")

    def finish(self):
        self.callback(self.points)
        self.destroy()
