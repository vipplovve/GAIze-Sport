import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path

class AnalysisConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_start_callback):
        super().__init__(parent)
        self.title("Analysis Configuration")
        self.geometry("550x260")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        self.on_start_callback = on_start_callback
        
        self.sport_var = ctk.StringVar(value="Football")
        self.yolo_path = ctk.StringVar(value="")
        self.lstm_path = ctk.StringVar(value="")
        self.esrgan_path = ctk.StringVar(value="")
        self.use_esrgan = ctk.BooleanVar(value=True)
        
        ctk.CTkLabel(self, text="⚙️ Configuration", font=("Roboto", 20, "bold"), text_color="#58a6ff").pack(pady=(20, 15))
        
        max_frame = ctk.CTkFrame(self, fg_color="transparent")
        max_frame.pack(fill="both", expand=True)

        sport_frame = ctk.CTkFrame(max_frame, fg_color="transparent")
        sport_frame.pack(fill="x", padx=40, pady=5)
        ctk.CTkLabel(sport_frame, text="Select Sport:", width=140, anchor="w", font=("Roboto", 13, "bold")).pack(side="left")
        ctk.CTkOptionMenu(sport_frame, variable=self.sport_var, values=["Football", "Basketball"], font=("Roboto", 12)).pack(side="left", fill="x", expand=True, padx=10)
        
        self.show_advanced = False
        self.toggle_btn = ctk.CTkButton(max_frame, text="Advanced Settings ▼", width=160, font=("Roboto", 12), fg_color="transparent", border_color="#555555", border_width=1, hover_color="#333333", command=self._toggle_advanced)
        self.toggle_btn.pack(pady=(15, 5))
        
        self.adv_frame = ctk.CTkFrame(max_frame, fg_color="transparent")
        ctk.CTkLabel(self.adv_frame, text="Leave paths blank to use default models for the selected sport.", font=("Roboto", 11, "italic"), text_color="#888888").pack(pady=(5, 5))
        self._build_row(self.adv_frame, "YOLO Model:", self.yolo_path)
        self._build_row(self.adv_frame, "LSTM Model:", self.lstm_path)
        self._build_row(self.adv_frame, "ESRGAN Model:", self.esrgan_path)
        
        chk_frame = ctk.CTkFrame(self.adv_frame, fg_color="transparent")
        chk_frame.pack(fill="x", padx=40, pady=5)
        ctk.CTkCheckBox(chk_frame, text="Upscale frames with ESRGAN 4x (Recommended)", variable=self.use_esrgan, font=("Roboto", 13)).pack(anchor="w")
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20, side="bottom")
        
        ctk.CTkButton(btn_frame, text="Start Video Analysis ▶", width=160, font=("Roboto", 14, "bold"), fg_color="#2cc985", hover_color="#25a870", command=self._start).pack(side="right", padx=40)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, font=("Roboto", 14), fg_color="#555555", hover_color="#777777", command=self.destroy).pack(side="right", padx=10)
        
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _toggle_advanced(self):
        self.show_advanced = not self.show_advanced
        if self.show_advanced:
            self.toggle_btn.configure(text="Advanced Settings ▲")
            self.adv_frame.pack(fill="both", expand=True, pady=5)
            self.geometry("550x480")
        else:
            self.toggle_btn.configure(text="Advanced Settings ▼")
            self.adv_frame.pack_forget()
            self.geometry("550x260")

    def _build_row(self, parent, label_text, string_var):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=40, pady=8)
        ctk.CTkLabel(row, text=label_text, width=120, anchor="w", font=("Roboto", 13, "bold")).pack(side="left")
        ctk.CTkEntry(row, textvariable=string_var, font=("Roboto", 12), placeholder_text="Default").pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(row, text="Browse", width=70, font=("Roboto", 12), command=lambda: self._browse(string_var)).pack(side="left")

    def _browse(self, string_var):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")])
        if path:
            string_var.set(path)
            
    def _start(self):
        yolo_val = self.yolo_path.get().strip()
        lstm_val = self.lstm_path.get().strip()
        esrgan_val = self.esrgan_path.get().strip()
        
        config = {
            "sport": self.sport_var.get(),
            "yolo_path": yolo_val if yolo_val else None,
            "lstm_path": lstm_val if lstm_val else None,
            "esrgan_path": esrgan_val if esrgan_val else None,
            "use_esrgan": self.use_esrgan.get()
        }
        self.on_start_callback(config)
        self.destroy()
