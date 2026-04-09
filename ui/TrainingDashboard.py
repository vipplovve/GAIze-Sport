import re
import time
from datetime import datetime
import customtkinter as ctk
import tkinter as tk

class MiniChart(tk.Canvas):
    def __init__(self, master, height=140, **kwargs):
        super().__init__(master, height=height, bg="#0d1117",
                         highlightthickness=0, bd=0, **kwargs)
        self.series = {}
        self.padding = {"left": 55, "right": 15, "top": 15, "bottom": 25}
        self.bind("<Configure>", lambda e: self._redraw())

    def add_series(self, name, color="#2cc985"):
        self.series[name] = {"color": color, "points": []}

    def add_point(self, series_name, x, y):
        if series_name in self.series:
            self.series[series_name]["points"].append((x, y))
            self._redraw()

    def clear_all(self):
        self.series = {}
        self.delete("all")
        self._draw_placeholder()

    def _draw_placeholder(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w > 1 and h > 1:
            self.create_text(w // 2, h // 2, text="Waiting for training data…",
                             fill="#333d4d", font=("Roboto", 11, "italic"))

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 120 or h < 60:
            return

        p = self.padding
        px, py = p["left"], p["top"]
        pw = w - p["left"] - p["right"]
        ph = h - p["top"] - p["bottom"]
        if pw <= 0 or ph <= 0:
            return

        all_x, all_y = [], []
        for s in self.series.values():
            for x, y in s["points"]:
                all_x.append(x)
                all_y.append(y)

        if not all_x:
            self._draw_placeholder()
            return

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            margin = max(abs(y_min) * 0.1, 0.001)
            y_min -= margin
            y_max += margin

        yr = y_max - y_min
        y_min -= yr * 0.05
        y_max += yr * 0.05

        def to_px(x, y):
            sx = px + (x - x_min) / (x_max - x_min) * pw
            sy = py + ph - (y - y_min) / (y_max - y_min) * ph
            return sx, sy

        for i in range(5):
            gy = py + i * ph / 4
            self.create_line(px, gy, px + pw, gy, fill="#161b22", dash=(2, 4))
            val = y_max - i * (y_max - y_min) / 4
            self.create_text(px - 6, gy, text=f"{val:.4g}", anchor="e",
                             fill="#484f58", font=("Consolas", 8))

        self.create_line(px, py + ph, px + pw, py + ph, fill="#21262d")
        self.create_line(px, py, px, py + ph, fill="#21262d")

        self.create_text(px + pw // 2, h - 6, text="Epoch",
                         fill="#484f58", font=("Roboto", 8))

        legend_x = px + pw - 5
        legend_y = py + 2
        for name, s in self.series.items():
            pts = s["points"]
            color = s["color"]

            if len(pts) >= 2:
                coords = []
                for x, y in pts:
                    cx, cy = to_px(x, y)
                    coords.extend([cx, cy])
                self.create_line(*coords, fill=color, width=2, smooth=True)

            if pts:
                lx, ly = pts[-1]
                cx, cy = to_px(lx, ly)
                r = 4
                self.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 fill=color, outline="#0d1117", width=1)
                self.create_text(cx + 8, cy, text=f"{ly:.4g}",
                                 anchor="w", fill=color, font=("Consolas", 8))

            self.create_rectangle(legend_x - 55, legend_y,
                                  legend_x - 43, legend_y + 8,
                                  fill=color, outline="")
            self.create_text(legend_x - 40, legend_y + 4, text=name,
                             anchor="w", fill="#8b949e", font=("Roboto", 8))
            legend_y += 15


class MetricCard(ctk.CTkFrame):
    def __init__(self, master, icon, label, value_color="#e6edf3", **kwargs):
        super().__init__(master, corner_radius=10, fg_color="#161b22",
                         border_width=1, border_color="#21262d", **kwargs)

        self.grid_columnconfigure(0, weight=1)

        self._header = ctk.CTkLabel(
            self, text=f"{icon}  {label}",
            font=("Roboto", 10), text_color="#6e7681", anchor="w"
        )
        self._header.grid(row=0, column=0, padx=10, pady=(7, 0), sticky="w")

        self._value = ctk.CTkLabel(
            self, text="—",
            font=("Consolas", 17, "bold"), text_color=value_color, anchor="w"
        )
        self._value.grid(row=1, column=0, padx=10, pady=(0, 7), sticky="w")

    def set_value(self, text, color=None):
        self._value.configure(text=str(text))
        if color:
            self._value.configure(text_color=color)

PRESET_ESRGAN = {
    "metrics": [
        {"key": "epoch", "icon": "📊", "label": "Epoch",    "color": "#e6edf3"},
        {"key": "loss",  "icon": "📉", "label": "Loss",     "color": "#f85149"},
        {"key": "main",  "icon": "📈", "label": "PSNR",     "color": "#3fb950"},
        {"key": "speed", "icon": "⚡",  "label": "Speed",    "color": "#d29922"},
        {"key": "eta",   "icon": "⏱",  "label": "ETA",      "color": "#58a6ff"},
    ],
    "chart": {"Loss": "#f85149"},
}

PRESET_LSTM = {
    "metrics": [
        {"key": "epoch", "icon": "📊", "label": "Epoch",    "color": "#e6edf3"},
        {"key": "loss",  "icon": "📉", "label": "Train Loss","color": "#f85149"},
        {"key": "main",  "icon": "🎯", "label": "Val Acc",   "color": "#3fb950"},
        {"key": "speed", "icon": "⚡",  "label": "Speed",    "color": "#d29922"},
        {"key": "eta",   "icon": "⏱",  "label": "ETA",      "color": "#58a6ff"},
    ],
    "chart": {"Train Loss": "#f85149", "Val Acc %": "#3fb950"},
}

PRESET_YOLO = {
    "metrics": [
        {"key": "epoch", "icon": "📊", "label": "Epoch",    "color": "#e6edf3"},
        {"key": "loss",  "icon": "📉", "label": "Loss",     "color": "#f85149"},
        {"key": "main",  "icon": "📈", "label": "Metric",   "color": "#3fb950"},
        {"key": "speed", "icon": "⚡",  "label": "Speed",    "color": "#d29922"},
        {"key": "eta",   "icon": "⏱",  "label": "ETA",      "color": "#58a6ff"},
    ],
    "chart": {"Loss": "#f85149"},
}

PRESET_ANALYSIS = {
    "metrics": [
        {"key": "frame",     "icon": "🎬", "label": "Frame",      "color": "#e6edf3"},
        {"key": "persons",   "icon": "👥", "label": "Persons",    "color": "#58a6ff"},
        {"key": "action",    "icon": "🏃", "label": "Action",     "color": "#3fb950"},
        {"key": "fps",       "icon": "⚡",  "label": "FPS",        "color": "#d29922"},
        {"key": "progress",  "icon": "📊", "label": "Progress",   "color": "#bc8cff"},
    ],
    "chart": {"Detections": "#58a6ff"},
}

class TrainingDashboard(ctk.CTkFrame):
    def __init__(self, master, preset=None, **kwargs):
        super().__init__(master, corner_radius=8, fg_color="transparent", **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._start_time = None
        self._current_epoch = 0
        self._total_epochs = 0
        self._chart_config = {}

        if preset is None:
            preset = PRESET_ESRGAN

        self._chart_config = preset.get("chart", {"Loss": "#f85149"})

        cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        cards_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 4))

        metric_defs = preset.get("metrics", PRESET_ESRGAN["metrics"])
        self._cards = {}
        for i, mc in enumerate(metric_defs):
            cards_frame.grid_columnconfigure(i, weight=1)
            card = MetricCard(cards_frame, icon=mc["icon"], label=mc["label"],
                              value_color=mc["color"])
            card.grid(row=0, column=i, padx=3, pady=0, sticky="ew")
            self._cards[mc["key"]] = card

        chart_border = ctk.CTkFrame(self, corner_radius=10, fg_color="#0d1117",
                                    border_width=1, border_color="#21262d")
        chart_border.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 4))

        self._chart = MiniChart(chart_border, height=145)
        self._chart.pack(fill="both", expand=True, padx=3, pady=3)
        self._init_chart_series()

        log_border = ctk.CTkFrame(self, corner_radius=10, fg_color="#0d1117",
                                  border_width=1, border_color="#21262d")
        log_border.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)

        self._log = ctk.CTkTextbox(
            log_border, font=("Consolas", 10),
            fg_color="#0d1117", text_color="#8b949e",
            state="disabled", wrap="word"
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

    def reset(self):
        self._start_time = time.time()
        self._current_epoch = 0
        self._total_epochs = 0

        for card in self._cards.values():
            card.set_value("—")

        self._chart.clear_all()
        self._init_chart_series()

        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def log(self, text):
        text = str(text)
        ts = datetime.now().strftime("%H:%M:%S")

        t_upper = text.upper()
        if "ERROR" in t_upper:
            ind = "🔴"
        elif "WARNING" in t_upper or "WARN" in t_upper:
            ind = "🟡"
        elif any(w in text.lower() for w in ("best", "complete", "success")):
            ind = "🟢"
        elif any(w in text.lower() for w in ("saved", "checkpoint", "loaded")):
            ind = "🔵"
        elif "epoch" in text.lower() and "/" in text:
            ind = "📶"
        else:
            ind = "  "

        line = f" {ind}  {ts}  │  {text}\n"

        def _append():
            self._log.configure(state="normal")
            self._log.insert("end", line)
            self._log.see("end")
            self._log.configure(state="disabled")

        try:
            self._log.after(0, _append)
        except Exception:
            pass

        self._parse(text)

    def set_metric(self, key, value, color=None):
        if key in self._cards:
            try:
                self._cards[key].after(
                    0, lambda: self._cards[key].set_value(value, color)
                )
            except Exception:
                pass

    def _init_chart_series(self):
        for name, color in self._chart_config.items():
            self._chart.add_series(name, color)

    def _add_chart_pt(self, series, x, y):
        try:
            self._chart.after(0, lambda: self._chart.add_point(series, x, y))
        except Exception:
            pass

    def _parse(self, text):
        m = re.search(r'Epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]', text)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            self._current_epoch = cur
            self._total_epochs = tot
            if self._start_time is None:
                self._start_time = time.time()

            self.set_metric("epoch", f"{cur} / {tot}")

            elapsed = time.time() - self._start_time
            if cur > 0:
                spd = elapsed / cur
                rem = (tot - cur) * spd
                self.set_metric("speed", f"{spd:.1f} s/ep")
                self.set_metric("eta", self._fmt_time(rem))

        m = re.search(r'L1 Loss:\s*([\d.]+)', text)
        if m:
            v = float(m.group(1))
            self.set_metric("loss", f"{v:.6f}")
            self._add_chart_pt("Loss", self._current_epoch, v)

        m = re.search(r'PSNR:\s*([\d.]+)', text)
        if m:
            self.set_metric("main", f"{float(m.group(1)):.2f} dB")

        m = re.search(r'G Loss:\s*([\d.]+)', text)
        if m:
            v = float(m.group(1))
            self.set_metric("loss", f"{v:.4f}")
            self._add_chart_pt("Loss", self._current_epoch, v)

        m = re.search(r'D Loss:\s*([\d.]+)', text)
        if m:
            self.set_metric("main", f"D: {float(m.group(1)):.4f}")

        m = re.search(r'Train Loss:\s*([\d.]+)', text)
        if m:
            v = float(m.group(1))
            self.set_metric("loss", f"{v:.4f}")
            self._add_chart_pt("Train Loss", self._current_epoch, v)

        m = re.search(r'Val Acc:\s*([\d.]+)%', text)
        if m:
            v = float(m.group(1))
            self.set_metric("main", f"{v:.1f}%")
            if "Val Acc %" not in self._chart.series:
                self._chart.add_series("Val Acc %", "#3fb950")
            self._add_chart_pt("Val Acc %", self._current_epoch, v)

        m = re.search(r'box_loss[=:\s]*([\d.]+)', text)
        if m:
            v = float(m.group(1))
            self.set_metric("loss", f"{v:.4f}")
            self._add_chart_pt("Loss", self._current_epoch, v)

        m = re.search(r'mAP50[=:\s]*([\d.]+)', text)
        if m:
            self.set_metric("main", f"mAP: {float(m.group(1)):.3f}")

        fm = re.search(r'F\s*(\d+)', text)
        pm = re.search(r'(\d+)P', text)
        if fm and pm:
            frame_num = int(fm.group(1))
            persons = int(pm.group(1))
            self.set_metric("frame", str(frame_num))
            self.set_metric("persons", str(persons))
            self._add_chart_pt("Detections", frame_num, persons)

        am = re.search(r'➜\s*(\w+)', text)
        if am:
            action = am.group(1)
            color = {"Sprinting": "#58a6ff", "Kicking": "#f85149",
                     "Dribbling": "#d29922", "Idle": "#6e7681"}.get(action, "#3fb950")
            self.set_metric("action", action, color)

    @staticmethod
    def _fmt_time(secs):
        if secs >= 3600:
            return f"{secs / 3600:.1f}h"
        elif secs >= 60:
            return f"{secs / 60:.1f}m"
        else:
            return f"{secs:.0f}s"
