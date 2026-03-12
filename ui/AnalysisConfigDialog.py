import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path

class AnalysisConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_start_callback):
        super().__init__(parent)
        self.title("Analysis Configuration")
        self.geometry("550x400")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        self.on_start_callback = on_start_callback
        
        base_dir = Path(__file__).resolve().parent.parent
        default_yolo = "yolo11n-pose.pt"
        default_lstm = str(base_dir / "training" / "lstm" / "checkpoints" / "action_lstm_best.pth")
        default_esrgan = str(base_dir / "training" / "esrgan" / "checkpoints" / "esrgan_generator.pth")
        
        self.yolo_path = ctk.StringVar(value=default_yolo)
        self.lstm_path = ctk.StringVar(value=default_lstm)
        self.esrgan_path = ctk.StringVar(value=default_esrgan)
        self.use_esrgan = ctk.BooleanVar(value=True)
        
        ctk.CTkLabel(self, text="⚙️ Model Configuration", font=("Roboto", 20, "bold"), text_color="#58a6ff").pack(pady=(20, 15))
        
        self._build_row("YOLOv11 Model:", self.yolo_path)
        self._build_row("LSTM Action Model:", self.lstm_path)
        self._build_row("ESRGAN Model:", self.esrgan_path)
        
        chk_frame = ctk.CTkFrame(self, fg_color="transparent")
        chk_frame.pack(fill="x", padx=40, pady=15)
        ctk.CTkCheckBox(chk_frame, text="Upscale frames with ESRGAN 4x (Recommended for low-res)", 
                        variable=self.use_esrgan, font=("Roboto", 13)).pack(anchor="w")
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20, side="bottom")
        
        ctk.CTkButton(btn_frame, text="Start Video Analysis ▶", width=160, font=("Roboto", 14, "bold"), 
                      fg_color="#2cc985", hover_color="#25a870", command=self._start).pack(side="right", padx=40)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, font=("Roboto", 14), 
                      fg_color="#555555", hover_color="#777777", command=self.destroy).pack(side="right", padx=10)
        
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build_row(self, label_text, string_var):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=40, pady=8)
        ctk.CTkLabel(row, text=label_text, width=140, anchor="w", font=("Roboto", 13, "bold")).pack(side="left")
        ctk.CTkEntry(row, textvariable=string_var, font=("Roboto", 12)).pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(row, text="Browse", width=70, font=("Roboto", 12), command=lambda: self._browse(string_var)).pack(side="left")

    def _browse(self, string_var):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")])
        if path:
            string_var.set(path)
            
    def _start(self):
        config = {
            "yolo_path": self.yolo_path.get(),
            "lstm_path": self.lstm_path.get(),
            "esrgan_path": self.esrgan_path.get(),
            "use_esrgan": self.use_esrgan.get()
        }
        self.on_start_callback(config)
        self.destroy()
