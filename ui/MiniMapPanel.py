import customtkinter as ctk
import tkinter as tk

class MiniMapPanel(ctk.CTkFrame):
    def __init__(self, master, width=300, height=200):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self.lbl_title = ctk.CTkLabel(self, text="2D Minimap", font=("Roboto", 14, "bold"))
        self.lbl_title.pack(pady=5)

        self.canvas = tk.Canvas(self, width=width-20, height=height-40, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(pady=5, padx=10)

        self.width = width - 20
        self.height = height - 40
        self.draw_pitch()

    def draw_pitch(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill="#2E8B57", outline="")

        w, h = self.width, self.height
        line_color = "white"

        self.canvas.create_rectangle(2, 2, w-2, h-2, outline=line_color, width=2)
        self.canvas.create_line(w/2, 0, w/2, h, fill=line_color, width=2)

        radius = h * 0.15
        self.canvas.create_oval(w/2 - radius, h/2 - radius, w/2 + radius, h/2 + radius, outline=line_color, width=2)

        pen_w = w * 0.15
        pen_h = h * 0.5
        self.canvas.create_rectangle(0, (h-pen_h)/2, pen_w, (h+pen_h)/2, outline=line_color, width=2)
        self.canvas.create_rectangle(w-pen_w, (h-pen_h)/2, w, (h+pen_h)/2, outline=line_color, width=2)

    def update_positions(self, player_positions, homography_matrix=None):
        self.draw_pitch()

        for p_id, cls_id, px, py in player_positions:
            map_x = px * self.width
            map_y = py * self.height

            color = "#FF4500"
            if px > 0.5:
                color = "#1E90FF"

            if cls_id == 0:
                color = "yellow"
                r = 3
            else:
                r = 4

            self.canvas.create_oval(map_x-r, map_y-r, map_x+r, map_y+r, fill=color, outline="black")
