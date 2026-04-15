import customtkinter as ctk
from .AppStyles import *
from .VideoPlayerWidget import VideoPlayerWidget
from .AIChatWidget import AIChatWidget
from .AnalyticsPanel import AnalyticsPanel
from .MiniMapPanel import MiniMapPanel
from .CalibrationView import CalibrationView
from .TrainingPanel import TrainingPanel
from .AnalysisConfigDialog import AnalysisConfigDialog
from .TrainingDashboard import TrainingDashboard, PRESET_ANALYSIS
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from pathlib import Path
import time


MODEL_DEFAULTS = {
    "Football": {
        "yolo": "models & metrics/football/yolo-finetuned/Football YOLO Fine-Tuned Model.pt",
        "lstm": "models & metrics/football/lstm/Football LSTM Action Recognition Model.pth",
        "esrgan": "models & metrics/football/esrgan/Football ESRGAN Phase #2 Generator.pth"
    },
    "Basketball": {
        "yolo": "models & metrics/basketball/yolo-finetuned/Basketball YOLO Fine-Tuned Model.pt",
        "lstm": "models & metrics/basketball/lstm/LSTM Basketball Action Recognition Model.pth",
        "esrgan": "models & metrics/basketball/esrgan/Basketball ESRGAN Phase #2 Generator.pth"
    }
}


class MainDashboard(ctk.CTkFrame):
    def __init__(self, master, on_logout=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_logout = on_logout

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="🛡️ GAIze-Sport",
                                        font=ctk.CTkFont(size=18, weight="bold"),
                                        text_color="#2cc985")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_upload = ctk.CTkButton(self.sidebar_frame, text="📁 Upload Video",
                                                    command=self.open_file_dialog)
        self.sidebar_button_upload.grid(row=1, column=0, padx=20, pady=8)

        self.sidebar_button_player = ctk.CTkButton(self.sidebar_frame, text="▶ Video Player",
                                                    command=self.show_player_view)
        self.sidebar_button_player.grid(row=2, column=0, padx=20, pady=8)

        self.sidebar_button_analysis = ctk.CTkButton(self.sidebar_frame, text="📊 Analysis",
                                                      command=self.show_analysis_view)
        self.sidebar_button_analysis.grid(row=3, column=0, padx=20, pady=8)

        self.sidebar_button_chat = ctk.CTkButton(self.sidebar_frame, text="💬 AI Chat",
                                                  command=self.show_chat_view)
        self.sidebar_button_chat.grid(row=4, column=0, padx=20, pady=8)

        self.sidebar_button_training = ctk.CTkButton(self.sidebar_frame, text="🧠 Model Training",
                                                      command=self.show_training_view,
                                                      fg_color="#6a1b9a", hover_color="#8e24aa")
        self.sidebar_button_training.grid(row=5, column=0, padx=20, pady=8)


        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="Status: Idle", text_color="gray",
                                        font=("Roboto", 11))
        self.lbl_status.grid(row=8, column=0, padx=20, pady=(5, 5), sticky="s")

        self.btn_logout = ctk.CTkButton(self.sidebar_frame, text="🔓 Logout", width=120,
                                         fg_color="#555555", hover_color="#777777",
                                         command=self._do_logout)
        self.btn_logout.grid(row=9, column=0, padx=20, pady=(5, 20), sticky="s")

        self.main_content = ctk.CTkFrame(self, corner_radius=10)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        self.video_player = None
        self.video_path = None
        self.analyzing = False
        self.analysis_data = {}
        self.action_counts = {}
        self.sport = "Football"
        self.total_analyzed_frames = 0
        self.analyzer = None
        self.keypoints_buffer = []

        self.player_frame = None
        self.analysis_results_frame = None
        self.training_frame = None
        self.chat_frame = None
        self.current_view = None

        self._build_player_view()
        self._build_training_view()
        self._build_chat_view()
        self._show_welcome()


    def _build_player_view(self):
        self.player_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.player_frame.grid_rowconfigure(0, weight=3)
        self.player_frame.grid_rowconfigure(1, weight=1)
        self.player_frame.grid_columnconfigure(0, weight=1)
        self.player_frame.grid_columnconfigure(1, weight=0)

        self.video_player = VideoPlayerWidget(self.player_frame)
        self.video_player.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        console_frame = ctk.CTkFrame(self.player_frame, corner_radius=8)
        console_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        console_frame.grid_rowconfigure(0, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)

        self.console_dashboard = TrainingDashboard(console_frame, preset=PRESET_ANALYSIS)
        self.console_dashboard.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        right_panel = ctk.CTkFrame(self.player_frame, width=280, fg_color="transparent")
        right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 5), pady=5)
        right_panel.grid_propagate(False)

        self.analysis_panel = AnalyticsPanel(right_panel, width=270, height=260)
        self.analysis_panel.pack(fill="x", pady=(0, 5))
        self.analysis_panel.btn_report.configure(command=self.generate_report)

        self.minimap = MiniMapPanel(right_panel, width=270, height=200)
        self.minimap.pack(fill="x", pady=5)

        self.btn_calibrate = ctk.CTkButton(right_panel, text="Calibrate Pitch",
                                            command=self.open_calibration)
        self.btn_calibrate.pack(pady=(5, 10))

    def _build_training_view(self):
        self.training_frame = TrainingPanel(self.main_content)

    def _build_chat_view(self):
        self.chat_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_widget = AIChatWidget(self.chat_frame)
        self.chat_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _show_welcome(self):
        self.welcome_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.welcome_frame.grid_rowconfigure(0, weight=1)
        self.welcome_frame.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(self.welcome_frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(inner, text="🛡️", font=("Arial", 48)).pack(pady=(0, 5))
        ctk.CTkLabel(inner, text="Welcome to GAIze-Sport",
                     font=("Roboto", 24, "bold"), text_color="#2cc985").pack(pady=(0, 10))
        ctk.CTkLabel(inner, text="Upload a video to get started, or explore the tools in the sidebar.",
                     font=("Roboto", 14), text_color="#a0a0a0").pack()

        ctk.CTkButton(inner, text="📁 Upload Video", width=180, height=40,
                      font=("Roboto", 14, "bold"),
                      fg_color="#2cc985", hover_color="#25a870",
                      command=self.open_file_dialog).pack(pady=(20, 0))

        self.welcome_frame.grid(row=0, column=0, sticky="nsew")
        self.current_view = "welcome"


    def _hide_all_views(self):
        for frame in [self.player_frame, self.analysis_results_frame,
                      self.training_frame, self.chat_frame]:
            if frame:
                frame.grid_forget()
        if hasattr(self, 'welcome_frame') and self.welcome_frame:
            self.welcome_frame.grid_forget()

    def show_player_view(self):
        self._hide_all_views()
        self.player_frame.grid(row=0, column=0, sticky="nsew")
        self.current_view = "player"

    def show_analysis_view(self):
        self._hide_all_views()

        if self.analysis_results_frame:
            self.analysis_results_frame.destroy()

        self.analysis_results_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.analysis_results_frame.grid(row=0, column=0, sticky="nsew")

        scroll = ctk.CTkScrollableFrame(self.analysis_results_frame)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(5, 10))
        ctk.CTkLabel(header, text="📊  Match Analysis Report",
                     font=("Roboto", 22, "bold"), text_color="#58a6ff").pack(side="left")
        if self.video_path:
            ctk.CTkLabel(header, text=f"  •  {Path(self.video_path).name}",
                         font=("Roboto", 13), text_color="#888888").pack(side="left", padx=(5, 0))
                         
        btn_report = ctk.CTkButton(header, text="📥 Download PDF Report", font=("Roboto", 13, "bold"),
                                   fg_color="#a83232", hover_color="#8b2828",
                                   command=self.generate_report)
        btn_report.pack(side="right", padx=10)

        if self.total_analyzed_frames == 0:
            empty = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=12)
            empty.pack(fill="x", pady=30, padx=20)
            ctk.CTkLabel(empty, text="📭", font=("Segoe UI Emoji", 40)).pack(pady=(20, 5))
            ctk.CTkLabel(empty, text="No Analysis Data Yet",
                         font=("Roboto", 18, "bold"), text_color="#ffffff").pack(pady=(0, 5))
            ctk.CTkLabel(empty, text="Upload and analyze a video to see detailed results here.",
                         font=("Roboto", 13), text_color="#888888").pack(pady=(0, 20))
            self.current_view = "analysis"
            return

        total_actions = sum(self.action_counts.values())

        cards_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        cards_frame.pack(fill="x", padx=5, pady=(0, 10))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        card_data = []
        card_data.append(("🎬", "Frames", str(self.total_analyzed_frames), "#2cc985"))

        avg_det = sum(d["detections"] for d in self.analysis_data.values()) / max(len(self.analysis_data), 1)
        card_data.append(("👥", "Avg Persons", f"{avg_det:.0f}", "#8ab4f8"))

        if total_actions > 0:
            dominant = max(self.action_counts, key=self.action_counts.get)
            dom_pct = int(self.action_counts[dominant] / total_actions * 100)
            card_data.append(("🏆", "Dominant", f"{dominant} ({dom_pct}%)", "#f0c040"))
        else:
            card_data.append(("🏆", "Dominant", "N/A", "#f0c040"))

        active = sum(v for k, v in self.action_counts.items() if k != "Idle")
        active_pct = int(active / total_actions * 100) if total_actions > 0 else 0
        card_data.append(("⚡", "Activity Rate", f"{active_pct}%", "#e06c75"))

        for col, (icon, title, value, color) in enumerate(card_data):
            card = ctk.CTkFrame(cards_frame, fg_color="#1a1a2e", corner_radius=10, border_width=1, border_color="#333")
            card.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(card, text=icon, font=("Segoe UI Emoji", 24)).pack(pady=(10, 2))
            ctk.CTkLabel(card, text=title, font=("Roboto", 11), text_color="#888888").pack()
            ctk.CTkLabel(card, text=value, font=("Roboto", 16, "bold"), text_color=color).pack(pady=(2, 10))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np

            chart_row = ctk.CTkFrame(scroll, fg_color="transparent")
            chart_row.pack(fill="x", padx=5, pady=5)
            chart_row.grid_columnconfigure((0, 1), weight=1)

            pie_frame = ctk.CTkFrame(chart_row, fg_color="#1a1a2e", corner_radius=10)
            pie_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(pie_frame, text="Action Distribution",
                         font=("Roboto", 13, "bold"), text_color="#58a6ff").pack(pady=(8, 0))

            fig1, ax1 = plt.subplots(figsize=(3.5, 2.8), facecolor="#1a1a2e")
            ax1.set_facecolor("#1a1a2e")
            labels = list(self.action_counts.keys())
            sizes = list(self.action_counts.values())
            colors_pie = ["#2cc985", "#8ab4f8", "#e06c75", "#f0c040", "#bc8cff"]
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_pie[:len(labels)],
                                                    autopct=lambda p: f'{int(p)}%' if p > 0 else '',
                                                    startangle=90, textprops={"color": "#cccccc", "fontsize": 10})
                for at in autotexts:
                    at.set_color("#ffffff")
                    at.set_fontweight("bold")
            else:
                ax1.text(0.5, 0.5, "No Data", ha="center", va="center", color="#666", fontsize=14,
                         transform=ax1.transAxes)
            fig1.tight_layout(pad=0.5)
            canvas1 = FigureCanvasTkAgg(fig1, master=pie_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(padx=5, pady=(0, 8))
            plt.close(fig1)

            bar_frame = ctk.CTkFrame(chart_row, fg_color="#1a1a2e", corner_radius=10)
            bar_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(bar_frame, text="Detection Density",
                         font=("Roboto", 13, "bold"), text_color="#58a6ff").pack(pady=(8, 0))

            fig2, ax2 = plt.subplots(figsize=(3.5, 2.8), facecolor="#1a1a2e")
            ax2.set_facecolor("#1a1a2e")
            sorted_keys = sorted(self.analysis_data.keys())
            if len(sorted_keys) > 0:
                num_bins = min(30, len(sorted_keys))
                bin_size = max(1, len(sorted_keys) // num_bins)
                bin_labels = []
                bin_values = []
                for b in range(0, len(sorted_keys), bin_size):
                    chunk = sorted_keys[b:b+bin_size]
                    avg = sum(self.analysis_data[k]["detections"] for k in chunk) / len(chunk)
                    bin_labels.append(chunk[0])
                    bin_values.append(avg)
                bars = ax2.bar(range(len(bin_values)), bin_values, color="#8ab4f8", edgecolor="#1a1a2e", width=0.8)
                ax2.set_xlabel("Frame Segment", color="#888", fontsize=9)
                ax2.set_ylabel("Avg Detections", color="#888", fontsize=9)
                ax2.tick_params(colors="#666", labelsize=8)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                ax2.spines["bottom"].set_color("#444")
                ax2.spines["left"].set_color("#444")
                ax2.set_xticks([])
            fig2.tight_layout(pad=0.5)
            canvas2 = FigureCanvasTkAgg(fig2, master=bar_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(padx=5, pady=(0, 8))
            plt.close(fig2)

            timeline_frame = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=10)
            timeline_frame.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(timeline_frame, text="Action Timeline",
                         font=("Roboto", 13, "bold"), text_color="#58a6ff").pack(pady=(8, 0))

            fig3, ax3 = plt.subplots(figsize=(8, 2.2), facecolor="#1a1a2e")
            ax3.set_facecolor("#1a1a2e")
            actions = list(self.action_counts.keys())
            colors_list = ["#2cc985", "#8ab4f8", "#e06c75", "#f0c040", "#bc8cff"]
            
            if len(actions) > 0:
                action_map = {a: i for i, a in enumerate(actions)}
                action_map["---"] = -0.5
                action_colors = {a: colors_list[i % len(colors_list)] for i, a in enumerate(actions)}
                action_colors["---"] = "#444444"
                ylabels = [a[:6] for a in actions]
            else:
                action_map = {"---": -0.5}
                action_colors = {"---": "#444444"}
                ylabels = []

            if sorted_keys:
                xs = sorted_keys
                ys = [action_map.get(self.analysis_data[k].get("action", "---"), -0.5) for k in xs]
                cs = [action_colors.get(self.analysis_data[k].get("action", "---"), "#444") for k in xs]
                ax3.scatter(xs, ys, c=cs, s=6, alpha=0.8)
                ax3.set_yticks(range(len(ylabels)))
                ax3.set_yticklabels(ylabels, color="#aaa", fontsize=9)
                ax3.set_xlabel("Frame Number", color="#888", fontsize=9)
                ax3.tick_params(axis="x", colors="#666", labelsize=8)
                ax3.spines["top"].set_visible(False)
                ax3.spines["right"].set_visible(False)
                ax3.spines["bottom"].set_color("#444")
                ax3.spines["left"].set_color("#444")
                ax3.set_ylim(-1, len(ylabels))
            fig3.tight_layout(pad=0.5)
            canvas3 = FigureCanvasTkAgg(fig3, master=timeline_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(padx=5, pady=(0, 8))
            plt.close(fig3)

        except ImportError:
            ctk.CTkLabel(scroll, text="Install matplotlib for charts: pip install matplotlib",
                         text_color="orange", font=("Roboto", 12)).pack(pady=10)

        assess_frame = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=10, border_width=1, border_color="#333")
        assess_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(assess_frame, text="🧠  AI Assessment Summary",
                     font=("Roboto", 14, "bold"), text_color="#f0c040").pack(anchor="w", padx=15, pady=(10, 5))

        assessment_lines = []
        if total_actions > 0:
            actions = list(self.action_counts.keys())
            pcts = {k: int(v / total_actions * 100) for k, v in self.action_counts.items()}

            assessment_lines.append(f"• Analyzed {self.total_analyzed_frames} frames with an average of {avg_det:.0f} person(s) detected per frame.")
            breakdown_str = ", ".join(f"{a} {pcts[a]}%" for a in actions)
            assessment_lines.append(f"• Action breakdown: {breakdown_str}.")

            idle_pct = pcts.get("Idle", 0)
            non_idle_pct = 100 - idle_pct
            if non_idle_pct >= 60:
                assessment_lines.append(f"• High activity ({non_idle_pct}% non-idle) — indicates an intense, fast-paced game.")
            elif non_idle_pct >= 30:
                assessment_lines.append(f"• Moderate activity — balanced movement patterns.")
            else:
                assessment_lines.append(f"• Low activity ({non_idle_pct}% non-idle) — limited player movement.")

            for a_name in actions:
                if a_name == "Idle":
                    continue
                if pcts.get(a_name, 0) > 20:
                    assessment_lines.append(f"• Significant {a_name.lower()} activity ({pcts[a_name]}%) — key moments captured.")

            if avg_det >= 5:
                assessment_lines.append(f"• Dense scene with ~{avg_det:.0f} persons/frame — good for tactical analysis.")
            elif avg_det >= 2:
                assessment_lines.append(f"• Moderate detection density (~{avg_det:.0f} persons/frame).")
            else:
                assessment_lines.append(f"• Sparse detections (~{avg_det:.0f} persons/frame) — consider using a wider camera angle.")
        else:
            assessment_lines.append("• Insufficient action recognition data for assessment. Ensure the video is long enough for LSTM analysis (≥30 frames of keypoints).")

        for line in assessment_lines:
            ctk.CTkLabel(assess_frame, text=line, font=("Roboto", 12),
                         text_color="#cccccc", wraplength=700, justify="left", anchor="w").pack(anchor="w", padx=15, pady=1)
        ctk.CTkLabel(assess_frame, text="", height=5).pack()

        breakdown_frame = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=10)
        breakdown_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(breakdown_frame, text="Action Breakdown",
                     font=("Roboto", 13, "bold"), text_color="#58a6ff").pack(anchor="w", padx=15, pady=(10, 5))

        actions = list(self.action_counts.keys())
        colors_list = ["#2cc985", "#8ab4f8", "#e06c75", "#f0c040", "#bc8cff"]
        bar_colors = {a: colors_list[i % len(colors_list)] for i, a in enumerate(actions)}
        for action, count in self.action_counts.items():
            pct = int(count / total_actions * 100) if total_actions > 0 else 0
            row = ctk.CTkFrame(breakdown_frame, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=3)
            ctk.CTkLabel(row, text=f"{action}", width=90, anchor="w",
                         font=("Roboto", 12, "bold"), text_color=bar_colors.get(action, "#ccc")).pack(side="left")
            bar = ctk.CTkProgressBar(row, width=250, height=14,
                                     progress_color=bar_colors.get(action, "#888"))
            bar.pack(side="left", padx=8)
            bar.set(pct / 100)
            ctk.CTkLabel(row, text=f"{count} frames  ({pct}%)",
                         font=("Roboto", 11), text_color="#aaaaaa").pack(side="left", padx=5)
        ctk.CTkLabel(breakdown_frame, text="", height=5).pack()

        if self.analysis_data:
            log_frame = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=10)
            log_frame.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(log_frame, text="Per-Frame Detection Log (last 60)",
                         font=("Roboto", 13, "bold"), text_color="#58a6ff").pack(anchor="w", padx=15, pady=(10, 5))

            log_box = ctk.CTkTextbox(log_frame, height=200, font=("Consolas", 11),
                                     fg_color="#0d1117", text_color="#58a6ff", state="disabled")
            log_box.pack(fill="x", padx=10, pady=(0, 10))

            log_box.configure(state="normal")
            sorted_frames = sorted(self.analysis_data.keys())[-60:]
            for fnum in sorted_frames:
                d = self.analysis_data[fnum]
                action = d.get("action", "---")
                conf = d.get("action_conf", 0)
                dets = d["detections"]
                line = f"▸ F{fnum:>5d} │ {dets}P │ {action:>10s} {int(conf*100):>3d}%"
                if d.get("det_details"):
                    line += f" │ {d['det_details']}"
                log_box.insert("end", line + "\n")
            log_box.configure(state="disabled")

        self.current_view = "analysis"

    def show_chat_view(self):
        self._hide_all_views()
        self.chat_frame.grid(row=0, column=0, sticky="nsew")
        self.current_view = "chat"

    def show_training_view(self):
        self._hide_all_views()
        self.training_frame.grid(row=0, column=0, sticky="nsew")
        self.current_view = "training"


    def log_to_console(self, text):
        if hasattr(self, 'console_dashboard') and self.console_dashboard:
            try:
                self.console_dashboard.log(text)
            except Exception:
                pass

    def _do_logout(self):
        if self.analyzing:
            self.analyzing = False
        if self.on_logout:
            self.on_logout()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path = file_path
            self.analysis_data = {}
            self.action_counts = {}
            self.total_analyzed_frames = 0
            self.show_player_view()
            self.video_player.load_video(file_path)
            self.lbl_status.configure(text="Status: Configuring Analysis...", text_color="yellow")
            
            AnalysisConfigDialog(self, self._on_config_done)

    def _on_config_done(self, config):
        self.analysis_config = config
        self.sport = config.get("sport", "Football")
        if self.sport == "Basketball":
            self.action_counts = {"Idle": 0, "Dribbling": 0, "Shooting": 0, "Guarding": 0}
        else:
            self.action_counts = {"Idle": 0, "Sprinting": 0, "Kicking": 0, "Dribbling": 0}
        self.lbl_status.configure(text="Status: Video Loaded", text_color="green")
        self.start_analysis()

    def open_calibration(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Load a video first before calibrating.")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            CalibrationView(self, frame, self.on_calibration_done)

    def on_calibration_done(self, points):
        self.log_to_console("[INFO] Pitch calibration updated!")


    def start_analysis(self):
        if not self.video_path:
            return

        self.analyzing = True
        self.lbl_status.configure(text="Status: Analyzing...", text_color="yellow")
        if hasattr(self, 'console_dashboard'):
            self.console_dashboard.reset()
        self.log_to_console("[INFO] Starting analysis...")
        threading.Thread(target=self.run_analysis_loop, daemon=True).start()

    def stop_analysis(self):
        self.analyzing = False
        self.lbl_status.configure(text="Status: Stopped", text_color="orange")
        self.log_to_console("[INFO] Analysis stopped by user.")

    def process_single_frame(self, frame):
        if self.analyzer is None:
            return frame
        try:
            import numpy as np
            import torch

            results = self.analyzer.model(frame, verbose=False)
            annotated = results[0].plot()

            if results[0].keypoints is not None:
                kpts = results[0].keypoints.xyn.cpu().numpy()
                if len(kpts) > 0:
                    self.keypoints_buffer.append(kpts[0])
                    if len(self.keypoints_buffer) > 30:
                        self.keypoints_buffer.pop(0)

                    if len(self.keypoints_buffer) >= 30:
                        input_data = np.array(self.keypoints_buffer).reshape(30, -1)
                        tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            out = self.analyzer.action_recognizer.model(tensor)
                            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

                        biased_probs = probs.copy()
                        if len(biased_probs) > 2:
                            biased_probs[2] += 0.05
                        if len(biased_probs) > 1:
                            biased_probs[1] += 0.03

                        action_labels = list(self.action_counts.keys())
                        if not action_labels:
                            action_labels = ["Idle", "Sprinting", "Kicking", "Dribbling"]
                        best_idx = int(biased_probs.argmax())
                        best_action = action_labels[best_idx]

                        y_pos = 40
                        cv2.putText(annotated, f"Action: {best_action}", (20, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        y_pos += 35
                        for i, label in enumerate(action_labels):
                            pct = int(probs[i] * 100)
                            color = (0, 255, 100) if i == best_idx else (180, 180, 180)
                            cv2.putText(annotated, f"  {label}: {pct}%", (20, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            y_pos += 28

            return annotated
        except Exception:
            return frame

    def run_analysis_loop(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.VideoAnalyticsEngine import VideoAnalyticsEngine
        import torch
        import numpy as np
        import os

        config = getattr(self, "analysis_config", {})
        sport = config.get("sport", "Football")
        project_root = Path(__file__).resolve().parent.parent
        
        defaults = MODEL_DEFAULTS.get(sport, MODEL_DEFAULTS["Football"])
        
        raw_yolo = config.get("yolo_path", "")
        yolo_path = raw_yolo if raw_yolo else str(project_root / defaults["yolo"])
        
        lstm_path = config.get("lstm_path", "")
        if not lstm_path:
            lstm_path = str(project_root / defaults["lstm"])
        
        use_esrgan = config.get("use_esrgan", False)
        raw_esrgan = config.get("esrgan_path", "")
        esrgan_path = raw_esrgan if raw_esrgan else str(project_root / defaults["esrgan"])
        
        self.log_to_console(f"[INFO] Loading YOLOv11 pose model ({Path(yolo_path).name}) for {sport}...")
        self.analyzer = VideoAnalyticsEngine(model_path=yolo_path, lstm_model=lstm_path, sport=sport)
        self.analyzer.load_model()
        self.keypoints_buffer = []
        
        esrgan_engine = None
        esrgan_out_dir = None
        if use_esrgan and esrgan_path:
            self.log_to_console(f"[INFO] Loading ESRGAN 4x model ({Path(esrgan_path).name})...")
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from core.FrameEnhancementEngine import FrameEnhancementEngine
            esrgan_engine = FrameEnhancementEngine(model_path=esrgan_path)
            esrgan_engine.load_model()
            if esrgan_engine.model is None:
                self.log_to_console("[WARNING] Failed to load ESRGAN. Falling back to normal.")
                use_esrgan = False
            else:
                esrgan_out_dir = Path(__file__).resolve().parent.parent / "videos" / "ESRGAN-enhanced"
                esrgan_out_dir.mkdir(parents=True, exist_ok=True)
                self.log_to_console(f"[INFO] Enhanced frames will be saved to: {esrgan_out_dir}")

        if self.video_player:
            self.video_player.frame_processor = self.process_single_frame

        self.log_to_console("[INFO] Models loaded. Processing frames...")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        loop_start = time.time()

        while cap.isOpened() and self.analyzing:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if use_esrgan and esrgan_engine:
                    enhanced_frame = esrgan_engine.enhance_frame(frame)
                    analysis_frame = enhanced_frame
                    
                    if esrgan_out_dir:
                        out_path = esrgan_out_dir / f"frame_{frame_count:05d}.jpg"
                        cv2.imwrite(str(out_path), enhanced_frame)
                else:
                    analysis_frame = frame

                results = self.analyzer.model(analysis_frame, verbose=False)
                annotated = results[0].plot()

                boxes = results[0].boxes
                num_detections = len(boxes) if boxes is not None else 0

                det_details = ""
                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    parts = []
                    for ci, cf in zip(classes, confs):
                        parts.append(f"cls{int(ci)}:{cf:.0%}")
                    det_details = " ".join(parts[:5])
                    if len(parts) > 5:
                        det_details += f" +{len(parts)-5} more"

                player_positions = []
                if boxes is not None and len(boxes) > 0:
                    frame_h, frame_w = analysis_frame.shape[:2]
                    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                        cx = ((box[0] + box[2]) / 2) / frame_w
                        cy = ((box[1] + box[3]) / 2) / frame_h
                        cls_id = int(classes[i]) if i < len(classes) else 0
                        player_positions.append((i, cls_id, float(cx), float(cy)))

                if player_positions:
                    try:
                        self.minimap.update_positions(player_positions)
                    except Exception:
                        pass

                action_name = "---"
                action_conf = 0.0
                all_probs = None

                if results[0].keypoints is not None:
                    kpts = results[0].keypoints.xyn.cpu().numpy()
                    if len(kpts) > 0:
                        self.keypoints_buffer.append(kpts[0])
                        if len(self.keypoints_buffer) > 30:
                            self.keypoints_buffer.pop(0)

                        if len(self.keypoints_buffer) >= 30:
                            input_data = np.array(self.keypoints_buffer).reshape(30, -1)
                            tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                            with torch.no_grad():
                                out = self.analyzer.action_recognizer.model(tensor)
                                probs = torch.softmax(out, dim=1)[0].cpu().numpy()

                            biased_probs = probs.copy()
                            if len(biased_probs) > 2:
                                biased_probs[2] += 0.05
                            if len(biased_probs) > 1:
                                biased_probs[1] += 0.03

                            action_labels = list(self.action_counts.keys())
                            if not action_labels:
                                action_labels = ["Idle", "Sprinting", "Kicking", "Dribbling"]
                            best_idx = int(biased_probs.argmax())
                            action_name = action_labels[best_idx]
                            action_conf = float(probs[best_idx])
                            all_probs = {label: int(probs[i] * 100) for i, label in enumerate(action_labels)}

                            y_pos = 40
                            cv2.putText(annotated, f"Action: {action_name}", (20, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            y_pos += 35
                            for i, label in enumerate(action_labels):
                                pct = int(probs[i] * 100)
                                color = (0, 255, 100) if i == best_idx else (180, 180, 180)
                                cv2.putText(annotated, f"  {label}: {pct}%", (20, y_pos),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                y_pos += 28

                            if action_name in self.action_counts:
                                self.action_counts[action_name] += 1

                cv2.putText(annotated, f"Persons: {num_detections}", (20, annotated.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                self.analysis_data[frame_count] = {
                    "detections": num_detections,
                    "action": action_name,
                    "action_conf": action_conf,
                    "det_details": det_details
                }
                self.total_analyzed_frames = frame_count + 1

                if self.video_player:
                    self.video_player.display_frame(annotated)
                    self.video_player.current_frame = frame_count
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.video_player.slider.set(progress)
                    self.video_player.lbl_frame_info.configure(text=f"{frame_count} / {total_frames}")

                if all_probs:
                    al = list(all_probs.keys())
                    action_str = f"F{frame_count}: {action_name} | " + " ".join([f"{a[:6]}:{all_probs[a]}%" for a in al]) + f" | {num_detections}P"
                else:
                    action_str = f"F{frame_count}: {action_name} | {num_detections} det"

                self.analysis_panel.update_stats({
                    "speed": round(num_detections * 3.5, 1),
                    "possession": {"Home": 55, "Away": 45},
                    "action": action_str
                })

                if all_probs:
                    al = list(all_probs.keys())
                    log_line = (f"▸ F{frame_count:>4d} │ {num_detections}P │ " +
                                " │ ".join([f"{a[:4]}:{all_probs[a]:>3d}%" for a in al]) +
                                f" │ ➜ {action_name}")
                else:
                    log_line = f"▸ F{frame_count:>4d} │ {num_detections}P │ {action_name}"
                if det_details:
                    log_line += f" │ {det_details}"
                self.log_to_console(log_line)

                elapsed_loop = time.time() - loop_start
                if elapsed_loop > 0 and frame_count > 0:
                    actual_fps = frame_count / elapsed_loop
                    try:
                        self.console_dashboard.set_metric("fps", f"{actual_fps:.1f}")
                    except Exception:
                        pass
                if total_frames > 0:
                    progress_pct = (frame_count / total_frames) * 100
                    try:
                        self.console_dashboard.set_metric("progress", f"{progress_pct:.1f}%")
                    except Exception:
                        pass

            except Exception as e:
                self.log_to_console(f"[ERROR] Frame {frame_count}: {e}")

            frame_count += 1
            delay = 1.0 / fps
            time.sleep(delay)

        cap.release()
        self.analyzing = False
        self.lbl_status.configure(text="Status: Analysis Complete", text_color="green")
        self.log_to_console(f"[DONE] Analyzed {frame_count} frames. Actions: {self.action_counts}")

    def generate_report(self):
        from tkinter import messagebox, filedialog
        import os
        
        if self.total_analyzed_frames == 0:
            messagebox.showwarning("No Data", "Analyze a video first before generating a report.")
            return

        default_name = f"{getattr(self, 'sport', 'Match')}_Analysis_Report.pdf"
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", 
                                                 initialfile=default_name,
                                                 title="Save PDF Report",
                                                 filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return

        self.lbl_status.configure(text="Status: Generating PDF...", text_color="blue")

        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from core.ReportGenerator import ReportGenerator

            out_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            gen = ReportGenerator(output_dir=out_dir)

            stats = {
                "action_counts": self.action_counts,
                "total_frames": self.total_analyzed_frames,
                "sport": getattr(self, 'sport', 'Football'),
                "analysis_data": self.analysis_data
            }
            video_name = Path(self.video_path).name if self.video_path else "Unknown"
            report_path = gen.generate_report(file_name, stats, video_name=video_name)

            messagebox.showinfo("Report Generated", f"PDF Report successfully saved to:\n{report_path}")
            self.lbl_status.configure(text="Status: Analysis Complete", text_color="green")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate PDF Report:\n{str(e)}")
            self.lbl_status.configure(text="Status: Report Generation Failed", text_color="red")
