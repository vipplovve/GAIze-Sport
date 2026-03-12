import customtkinter as ctk
from .AppStyles import *


class LoginScreen(ctk.CTkFrame):

    def __init__(self, master, on_login_success, **kwargs):
        super().__init__(master, **kwargs)
        self.on_login_success = on_login_success

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_icon = ctk.CTkLabel(center, text="🛡️", font=("Segoe UI Emoji", 60))
        self.lbl_icon.pack(pady=(0, 2))

        self.lbl_title = ctk.CTkLabel(center, text="GAIze-Sport",
                                       font=("Roboto", 44, "bold"),
                                       text_color="#2cc985")
        self.lbl_title.pack(pady=(0, 3))

        self.lbl_subtitle = ctk.CTkLabel(center, text="AI-Powered Sports Video Analysis Platform",
                                          font=("Roboto", 16, "bold"),
                                          text_color="#8ab4f8")
        self.lbl_subtitle.pack(pady=(0, 5))

        self.lbl_tagline = ctk.CTkLabel(center, text="Real-time detection  •  Super-resolution  •  Action recognition",
                                         font=("Roboto", 12),
                                         text_color="#888888")
        self.lbl_tagline.pack(pady=(0, 15))

        desc_frame = ctk.CTkFrame(center, fg_color="#1a1a2e", corner_radius=12, border_width=1, border_color="#2cc985")
        desc_frame.pack(fill="x", padx=25, pady=(0, 15))

        features = [
            ("🎯", "YOLOv11 Pose Detection", "Track players and detect body keypoints in real-time"),
            ("✨", "ESRGAN Super-Resolution", "Upscale low-quality frames to stunning 4× clarity"),
            ("🏃", "LSTM Action Recognition", "Classify sprinting, kicking, and idle actions live"),
            ("🗺️", "2D Tactical Minimap", "Bird's-eye view of all player positions on pitch"),
            ("📄", "PDF Match Reports", "Generate detailed analysis reports with one click"),
        ]

        for icon, title, desc in features:
            row_frame = ctk.CTkFrame(desc_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=16, pady=3)

            ctk.CTkLabel(row_frame, text=icon, font=("Segoe UI Emoji", 16), width=26).pack(side="left", padx=(0, 7))

            text_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            text_frame.pack(side="left", fill="x")

            ctk.CTkLabel(text_frame, text=f"{title} — {desc}", font=("Roboto", 12),
                         text_color="#cccccc", anchor="w").pack(anchor="w")

        ctk.CTkLabel(desc_frame, text="", height=3).pack()

        form_frame = ctk.CTkFrame(center, fg_color="transparent")
        form_frame.pack(pady=(0, 10))

        ctk.CTkLabel(form_frame, text="Username", font=("Roboto", 12, "bold"),
                     text_color="#bbbbbb").grid(row=0, column=0, sticky="w", padx=5, pady=(0, 2))
        self.entry_user = ctk.CTkEntry(form_frame, width=290, height=38,
                                        font=("Roboto", 13),
                                        placeholder_text="Enter your username",
                                        corner_radius=9)
        self.entry_user.grid(row=1, column=0, padx=5, pady=(0, 10))

        ctk.CTkLabel(form_frame, text="Password", font=("Roboto", 12, "bold"),
                     text_color="#bbbbbb").grid(row=2, column=0, sticky="w", padx=5, pady=(0, 2))
        self.entry_pass = ctk.CTkEntry(form_frame, width=290, height=38,
                                        font=("Roboto", 13),
                                        placeholder_text="Enter your password",
                                        show="•",
                                        corner_radius=9)
        self.entry_pass.grid(row=3, column=0, padx=5, pady=(0, 10))

        self.entry_pass.bind("<Return>", lambda e: self._do_login())

        self.lbl_error = ctk.CTkLabel(form_frame, text="", font=("Roboto", 11, "bold"),
                                       text_color="#ff5555")
        self.lbl_error.grid(row=4, column=0, pady=(0, 5))

        btn_frame = ctk.CTkFrame(center, fg_color="transparent")
        btn_frame.pack(pady=(0, 10))

        self.btn_login = ctk.CTkButton(btn_frame, text="🔐  Login", width=145, height=42,
                                        font=("Roboto", 14, "bold"),
                                        fg_color="#2cc985", hover_color="#25a870",
                                        corner_radius=10,
                                        command=self._do_login)
        self.btn_login.pack(side="left", padx=10)

        self.btn_quit = ctk.CTkButton(btn_frame, text="✕  Quit", width=145, height=42,
                                       font=("Roboto", 14, "bold"),
                                       fg_color="#c0392b", hover_color="#e74c3c",
                                       corner_radius=10,
                                       command=self._do_quit)
        self.btn_quit.pack(side="left", padx=10)

        libs_list = "PyTorch • Ultralytics • OpenCV • CustomTkinter • NumPy • Pandas • Matplotlib • Transformers • ReportLab • yt-dlp • Pillow"
        self.lbl_footer = ctk.CTkLabel(center, text=f"© 2026 GAIze-Sport  •  Major Project II\nPowered by: {libs_list}",
                                        font=("Roboto", 10),
                                        text_color="#555555")
        self.lbl_footer.pack(pady=(15, 5))

    def _do_login(self):
        username = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()

        if not username or not password:
            self.lbl_error.configure(text="⚠  Please enter both username and password.")
            return

        if (username == "viplove") and (password == "skrrtskrrt"):
            self.on_login_success(username)
        else:
            self.lbl_error.configure(text="⚠  Invalid credentials. Please try again.")

    def _do_quit(self):
        self.winfo_toplevel().destroy()
