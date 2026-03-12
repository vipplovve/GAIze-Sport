import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(PROJECT_ROOT_PATH)

for _subdir in ["esrgan", "lstm", "yolo"]:
    _p = str(PROJECT_ROOT_PATH / "training" / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TrainingPanel(ctk.CTkFrame):

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.stop_event = threading.Event()
        self.training_thread = None

        self.tabview = ctk.CTkTabview(self, corner_radius=8)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.tabview.add("ESRGAN")
        self.tabview.add("LSTM")
        self.tabview.add("YOLO Fine-tune")

        self._build_esrgan_tab()
        self._build_lstm_tab()
        self._build_yolo_tab()


    def _make_field(self, parent, label, default, row, browse=False, browse_dir=False):
        lbl = ctk.CTkLabel(parent, text=label, font=("Roboto", 12), anchor="w", width=130)
        lbl.grid(row=row, column=0, padx=(10, 5), pady=4, sticky="w")

        entry = ctk.CTkEntry(parent, width=250, font=("Consolas", 12))
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=4, sticky="ew")

        if browse or browse_dir:
            def do_browse():
                if browse_dir:
                    path = filedialog.askdirectory()
                else:
                    path = filedialog.askopenfilename()
                if path:
                    entry.delete(0, "end")
                    entry.insert(0, path)

            btn = ctk.CTkButton(parent, text="📁", width=35, command=do_browse)
            btn.grid(row=row, column=2, padx=(0, 10), pady=4)

        return entry

    def _make_controls(self, parent, start_cmd, row):
        ctrl_frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 5))

        btn_start = ctk.CTkButton(ctrl_frame, text="▶ Start Training", width=150,
                                  fg_color="#2cc985", hover_color="#25a870",
                                  font=("Roboto", 13, "bold"), command=start_cmd)
        btn_start.pack(side="left", padx=5)

        btn_stop = ctk.CTkButton(ctrl_frame, text="⬛ Stop", width=80,
                                 fg_color="#8B0000", hover_color="#B22222",
                                 font=("Roboto", 13), command=self._stop_training)
        btn_stop.pack(side="left", padx=5)

        progress = ctk.CTkProgressBar(ctrl_frame, width=200)
        progress.pack(side="left", padx=10, fill="x", expand=True)
        progress.set(0)

        status_lbl = ctk.CTkLabel(ctrl_frame, text="Idle", font=("Roboto", 11), text_color="gray")
        status_lbl.pack(side="right", padx=10)

        return btn_start, btn_stop, progress, status_lbl

    def _make_log(self, parent, row):
        log_frame = ctk.CTkFrame(parent, corner_radius=8)
        log_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=10, pady=(5, 10))

        lbl = ctk.CTkLabel(log_frame, text="Training Log", font=("Roboto", 12, "bold"))
        lbl.pack(anchor="w", padx=10, pady=(5, 0))

        log_box = ctk.CTkTextbox(log_frame, height=200, font=("Consolas", 10),
                                 fg_color="#1a1a2e", text_color="#00ff88",
                                 state="disabled", wrap="word")
        log_box.pack(fill="both", expand=True, padx=5, pady=5)

        return log_box

    def _log_to(self, log_box, text):
        def _append():
            log_box.configure(state="normal")
            log_box.insert("end", str(text) + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")
        try:
            log_box.after(0, _append)
        except Exception:
            pass

    def _stop_training(self):
        self.stop_event.set()

    def _is_training(self):
        return self.training_thread is not None and self.training_thread.is_alive()


    def _build_esrgan_tab(self):
        tab = self.tabview.tab("ESRGAN")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(tab, text="ESRGAN Super-Resolution Training",
                     font=("Roboto", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(10, 5))

        esrgan_ckpt_dir = str(PROJECT_ROOT_PATH / "training" / "esrgan" / "checkpoints")
        esrgan_hr_dir = str(PROJECT_ROOT_PATH / "training" / "data" / "hr_frames")

        self.esrgan_phase = self._make_field(tab, "Phase (1=PSNR, 2=GAN):", "1", 1)
        self.esrgan_hr_dir = self._make_field(tab, "HR Images Dir:", esrgan_hr_dir, 2, browse_dir=True)
        self.esrgan_epochs = self._make_field(tab, "Epochs:", "50", 3)
        self.esrgan_batch = self._make_field(tab, "Batch Size:", "4", 4)
        self.esrgan_lr = self._make_field(tab, "Learning Rate:", "0.0002", 5)
        self.esrgan_crop = self._make_field(tab, "HR Crop Size:", "128", 6)
        self.esrgan_blocks = self._make_field(tab, "RRDB Blocks:", "8", 7)
        self.esrgan_pretrained = self._make_field(tab, "Pretrained Gen:", "", 8, browse=True)
        self.esrgan_ckpt = self._make_field(tab, "Checkpoint Dir:", esrgan_ckpt_dir, 9, browse_dir=True)

        self.esrgan_start, self.esrgan_stop_btn, self.esrgan_progress, self.esrgan_status = \
            self._make_controls(tab, self._start_esrgan, 10)

        self.btn_view_esrgan_metrics = ctk.CTkButton(tab, text="📊 View GAN Metrics", 
                                                     command=self._view_esrgan_metrics,
                                                     fg_color="#3b5998", hover_color="#2d4373")
        self.btn_view_esrgan_metrics.grid(row=11, column=0, columnspan=3, pady=(5, 5))

        self.esrgan_log = self._make_log(tab, 12)

    def _start_esrgan(self):
        if self._is_training():
            messagebox.showwarning("Busy", "Another training session is already running.")
            return

        config = {
            'phase': int(self.esrgan_phase.get()),
            'hr_dir': self.esrgan_hr_dir.get(),
            'epochs': int(self.esrgan_epochs.get()),
            'batch_size': int(self.esrgan_batch.get()),
            'lr': float(self.esrgan_lr.get()),
            'hr_crop': int(self.esrgan_crop.get()),
            'num_blocks': int(self.esrgan_blocks.get()),
            'pretrained_gen': self.esrgan_pretrained.get() or None,
            'checkpoint_dir': self.esrgan_ckpt.get(),
            'save_every': 5,
            'num_workers': 0,
        }

        self.stop_event.clear()
        self.esrgan_progress.set(0)
        self.esrgan_status.configure(text="Training...", text_color="yellow")

        def run():
            try:
                sys.path.insert(0, str(PROJECT_ROOT_PATH / "training" / "esrgan"))
                from training.esrgan.TrainESRGAN import train_from_gui

                train_from_gui(config,
                               log_callback=lambda t: self._log_to(self.esrgan_log, t),
                               stop_event=self.stop_event)

                self.esrgan_status.configure(text="Complete ✓", text_color="#2cc985")
                self.esrgan_progress.set(1.0)
            except Exception as e:
                self._log_to(self.esrgan_log, f"ERROR: {e}")
                self.esrgan_status.configure(text="Failed ✗", text_color="red")

        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def _view_esrgan_metrics(self):
        ckpt_dir = self.esrgan_ckpt.get()
        paths = [
            os.path.join(ckpt_dir, "esrgan_phase1_loss.png"),
            os.path.join(ckpt_dir, "esrgan_phase1_psnr.png"),
            os.path.join(ckpt_dir, "esrgan_gan_loss.png")
        ]
        self._show_metrics_window("ESRGAN Metrics", paths)


    def _build_lstm_tab(self):
        tab = self.tabview.tab("LSTM")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(9, weight=1)

        ctk.CTkLabel(tab, text="LSTM Action Recognition Training",
                     font=("Roboto", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(10, 5))

        lstm_ckpt_dir = str(PROJECT_ROOT_PATH / "training" / "lstm" / "checkpoints")
        lstm_data_dir = str(PROJECT_ROOT_PATH / "training" / "lstm" / "data")

        self.lstm_data_dir = self._make_field(tab, "Data Directory:", lstm_data_dir, 1, browse_dir=True)
        self.lstm_source_clips = self._make_field(tab, "Clips Source Dir:", "", 2, browse_dir=True)
        self.lstm_epochs = self._make_field(tab, "Epochs:", "30", 3)
        self.lstm_batch = self._make_field(tab, "Batch Size:", "32", 4)
        self.lstm_lr = self._make_field(tab, "Learning Rate:", "0.001", 5)
        self.lstm_input = self._make_field(tab, "Input Size:", "34", 6)
        self.lstm_hidden = self._make_field(tab, "Hidden Size:", "64", 7)
        self.lstm_layers = self._make_field(tab, "LSTM Layers:", "2", 8)
        self.lstm_classes = self._make_field(tab, "Num Classes:", "3", 9)
        self.lstm_ckpt = self._make_field(tab, "Checkpoint Dir:", lstm_ckpt_dir, 10, browse_dir=True)

        self.lstm_start, self.lstm_stop_btn, self.lstm_progress, self.lstm_status = \
            self._make_controls(tab, self._start_lstm, 11)
            
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=12, column=0, columnspan=3, pady=5)

        self.btn_extract_lstm_data = ctk.CTkButton(btn_frame, text="⚙ Extract Dataset from Clips", 
                                                   command=self._start_lstm_extraction,
                                                   fg_color="#a86a25", hover_color="#8a5318")
        self.btn_extract_lstm_data.pack(side="left", padx=5)

        self.btn_generate_synthetic = ctk.CTkButton(btn_frame, text="🧪 Generate Synthetic Data", 
                                                    command=self._start_synthetic_generation,
                                                    fg_color="#800080", hover_color="#4B0082")
        self.btn_generate_synthetic.pack(side="left", padx=5)

        self.btn_view_lstm_metrics = ctk.CTkButton(btn_frame, text="📊 View LSTM Metrics", 
                                                   command=self._view_lstm_metrics,
                                                   fg_color="#3b5998", hover_color="#2d4373")
        self.btn_view_lstm_metrics.pack(side="left", padx=5)

        self.lstm_log = self._make_log(tab, 13)

    def _start_lstm(self):
        if self._is_training():
            messagebox.showwarning("Busy", "Another training session is already running.")
            return

        config = {
            'data_dir': self.lstm_data_dir.get(),
            'epochs': int(self.lstm_epochs.get()),
            'batch_size': int(self.lstm_batch.get()),
            'lr': float(self.lstm_lr.get()),
            'input_size': int(self.lstm_input.get()),
            'hidden_size': int(self.lstm_hidden.get()),
            'num_layers': int(self.lstm_layers.get()),
            'num_classes': int(self.lstm_classes.get()),
            'checkpoint_dir': self.lstm_ckpt.get(),
        }

        self.stop_event.clear()
        self.lstm_progress.set(0)
        self.lstm_status.configure(text="Training...", text_color="yellow")

        def run():
            try:
                from training.lstm.TrainLSTM import train_from_gui

                train_from_gui(config,
                               log_callback=lambda t: self._log_to(self.lstm_log, t),
                               stop_event=self.stop_event)

                self.lstm_status.configure(text="Complete ✓", text_color="#2cc985")
                self.lstm_progress.set(1.0)
            except Exception as e:
                self._log_to(self.lstm_log, f"ERROR: {e}")
                self.lstm_status.configure(text="Failed ✗", text_color="red")

        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def _start_lstm_extraction(self):
        if self._is_training():
            messagebox.showwarning("Busy", "Another session is already running.")
            return

        source_dir = self.lstm_source_clips.get()
        if not source_dir or not os.path.exists(source_dir):
            messagebox.showerror("Error", "Please select a valid Clips Source Directory first.")
            return

        out_dir = self.lstm_data_dir.get()
        
        self.stop_event.clear()
        self.lstm_progress.set(0)
        self.lstm_status.configure(text="Extracting...", text_color="orange")
        
        def run():
            try:
                from training.lstm.ExtractRealLSTMData import extract_lstm_data_from_videos
                success = extract_lstm_data_from_videos(
                    source_dir=source_dir,
                    output_dir=out_dir,
                    seq_len=30,
                    log_callback=lambda t: self._log_to(self.lstm_log, t),
                    stop_event=self.stop_event
                )
                if success:
                    self.lstm_status.configure(text="Extraction Complete ✓", text_color="#2cc985")
                    self.lstm_progress.set(1.0)
                else:
                    self.lstm_status.configure(text="Extraction Failed/Stopped", text_color="red")
            except Exception as e:
                self._log_to(self.lstm_log, f"ERROR: {e}")
                self.lstm_status.configure(text="Failed ✗", text_color="red")

        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def _start_synthetic_generation(self):
        if self._is_training():
            messagebox.showwarning("Busy", "Another session is already running.")
            return

        out_dir = self.lstm_data_dir.get()
        if not out_dir or not os.path.exists(out_dir):
            messagebox.showerror("Error", "Please select a valid Data Directory first.")
            return

        self.stop_event.clear()
        self.lstm_progress.set(0)
        self.lstm_status.configure(text="Generating...", text_color="orange")
        
        def run():
            try:
                sys.path.insert(0, str(PROJECT_ROOT_PATH / "training"))
                import PrepareTrainingData
                self._log_to(self.lstm_log, "Generating 100 synthetic sequences per class...")
                PrepareTrainingData.generate_lstm_data(samples_per_class=100)
                self._log_to(self.lstm_log, f"Synthetic data generation complete! Data saved to {out_dir}")
                
                self.lstm_status.configure(text="Generation Complete ✓", text_color="#2cc985")
                self.lstm_progress.set(1.0)
            except Exception as e:
                self._log_to(self.lstm_log, f"ERROR: {e}")
                self.lstm_status.configure(text="Failed ✗", text_color="red")
        
        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def _view_lstm_metrics(self):
        ckpt_dir = self.lstm_ckpt.get()
        paths = [
            os.path.join(ckpt_dir, "lstm_loss.png"),
            os.path.join(ckpt_dir, "lstm_accuracy.png"),
            os.path.join(ckpt_dir, "lstm_confusion_matrix.png")
        ]
        self._show_metrics_window("LSTM Metrics", paths)


    def _build_yolo_tab(self):
        tab = self.tabview.tab("YOLO Fine-tune")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(tab, text="YOLOv11 Pose Fine-tuning",
                     font=("Roboto", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(10, 5))

        yolo_model = str(PROJECT_ROOT_PATH / "training" / "yolo11n-pose.pt")

        self.yolo_model = self._make_field(tab, "Base Model:", yolo_model, 1, browse=True)
        self.yolo_data = self._make_field(tab, "Dataset YAML:", "", 2, browse=True)
        self.yolo_epochs = self._make_field(tab, "Epochs:", "50", 3)
        self.yolo_imgsz = self._make_field(tab, "Image Size:", "640", 4)
        self.yolo_batch = self._make_field(tab, "Batch Size:", "8", 5)
        self.yolo_device = self._make_field(tab, "Device (cpu/0):", "cpu", 6)

        self.yolo_start, self.yolo_stop_btn, self.yolo_progress, self.yolo_status = \
            self._make_controls(tab, self._start_yolo, 7)

        self.yolo_log = self._make_log(tab, 8)

    def _start_yolo(self):
        if self._is_training():
            messagebox.showwarning("Busy", "Another training session is already running.")
            return

        config = {
            'base_model': self.yolo_model.get(),
            'data_yaml': self.yolo_data.get(),
            'epochs': int(self.yolo_epochs.get()),
            'imgsz': int(self.yolo_imgsz.get()),
            'batch': int(self.yolo_batch.get()),
            'device': self.yolo_device.get(),
            'project': str(PROJECT_ROOT_PATH / "training" / "runs" / "train"),
            'name': 'yolo_finetune',
        }

        self.stop_event.clear()
        self.yolo_progress.set(0)
        self.yolo_status.configure(text="Training...", text_color="yellow")

        def run():
            try:
                from training.yolo.TrainYOLO import train_from_gui

                train_from_gui(config,
                               log_callback=lambda t: self._log_to(self.yolo_log, t),
                               stop_event=self.stop_event)

                self.yolo_status.configure(text="Complete ✓", text_color="#2cc985")
                self.yolo_progress.set(1.0)
            except Exception as e:
                self._log_to(self.yolo_log, f"ERROR: {e}")
                self.yolo_status.configure(text="Failed ✗", text_color="red")

        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()


    def _show_metrics_window(self, title, image_paths):
        top = ctk.CTkToplevel(self)
        top.title(title)
        top.geometry("900x700")
        
        top.attributes('-topmost', True)
        top.after(100, lambda: top.attributes('-topmost', False))
        
        scroll = ctk.CTkScrollableFrame(top)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        from PIL import Image
        
        found_any = False
        for path in image_paths:
            if os.path.exists(path):
                found_any = True
                try:
                    img = Image.open(path)
                    max_width = 800
                    if img.width > max_width:
                        ratio = max_width / img.width
                        new_h = int(img.height * ratio)
                        img = img.resize((max_width, new_h), Image.LANCZOS)
                        
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                    lbl = ctk.CTkLabel(scroll, text="", image=ctk_img)
                    lbl.pack(pady=10)
                    
                    title_lbl = ctk.CTkLabel(scroll, text=os.path.basename(path), font=("Roboto", 14, "bold"))
                    title_lbl.pack(pady=(0, 20))
                except Exception as e:
                    err_lbl = ctk.CTkLabel(scroll, text=f"Error loading {os.path.basename(path)}: {e}", text_color="red")
                    err_lbl.pack(pady=10)
                    
        if not found_any:
            msg = ctk.CTkLabel(scroll, text="No metrics plots found.\nPlease run training first or check the checkpoints directory.", 
                               font=("Roboto", 16), text_color="orange")
            msg.pack(pady=50)
