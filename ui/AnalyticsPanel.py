import customtkinter as ctk

class AnalyticsPanel(ctk.CTkFrame):
    def __init__(self, master, width=300, height=300):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self.lbl_title = ctk.CTkLabel(self, text="Match Analysis", font=("Roboto", 16, "bold"))
        self.lbl_title.pack(pady=10)

        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_speed = ctk.CTkLabel(self.stats_frame, text="Avg Speed: -- km/h", font=("Roboto", 12))
        self.lbl_speed.grid(row=0, column=0, padx=5, sticky="w")

        self.lbl_possession = ctk.CTkLabel(self.stats_frame, text="Possession: 50% - 50%", font=("Roboto", 12))
        self.lbl_possession.grid(row=1, column=0, padx=5, sticky="w")

        self.lbl_class = ctk.CTkLabel(self, text="Teams detected: --", text_color="gray", font=("Roboto", 12))
        self.lbl_class.pack(pady=(10, 5))

        self.lbl_actions = ctk.CTkLabel(self, text="Action Log", font=("Roboto", 14, "bold"))
        self.lbl_actions.pack(pady=(15, 5))

        self.log_box = ctk.CTkTextbox(self, height=120, font=("Consolas", 10),
                                       fg_color="#0d1117", text_color="#58a6ff",
                                       state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)

        self._log_line_count = 0
        self._max_lines = 30

        self.btn_report = ctk.CTkButton(self, text="Generate PDF Report", fg_color="#2E8B57", hover_color="#3CB371", command=self.generate_report)
        self.btn_report.pack(pady=10, padx=10, fill="x")

    def update_stats(self, stats_dict):
        if "speed" in stats_dict:
            self.lbl_speed.configure(text=f"Avg Speed: {stats_dict['speed']:.1f} km/h")

        if "possession" in stats_dict:
            p = stats_dict.get("possession", None)
            if p:
                self.lbl_possession.configure(text=f"Possession: H {p.get('Home', 0)}% - A {p.get('Away', 0)}%")

        if "action" in stats_dict:
            self.add_log_entry(stats_dict["action"])

    def add_log_entry(self, text):
        if not text:
            return
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"• {text}\n")
            self._log_line_count += 1
            if self._log_line_count > self._max_lines:
                self.log_box.delete("1.0", "2.0")
                self._log_line_count -= 1
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def generate_report(self):
        if hasattr(self.master, 'generate_report'):
            self.master.generate_report()
        else:
            try:
                self.master.master.generate_report()
            except Exception:
                pass
