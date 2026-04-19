import customtkinter as ctk
import threading
import json
import urllib.request
import urllib.error
import time
import tkinter as tk
import os

OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}/api/chat"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a sports video analysis assistant embedded in GAIze-Sport, an AI-powered platform "
    "for analysing football and basketball match footage. The platform uses YOLOv11 for pose "
    "detection, an LSTM model for action recognition (Idle, Sprinting, Kicking, Dribbling for "
    "football; Idle, Dribbling, Shooting, Guarding for basketball), and ESRGAN for frame "
    "super-resolution.\n\n"
    "Answer the user's questions about the current analysis session, sports analytics in general, "
    "or the AI models used. Be concise and helpful. If analysis context is available below, use "
    "it to give data-driven answers.\n"
)

COLORS = {
    "bg_primary": "#0b0f19",
    "bg_secondary": "#111827",
    "bg_tertiary": "#1e293b",
    "surface": "#1e293b",
    "surface_hover": "#273548",
    "border": "#334155",
    "border_focus": "#6366f1",
    "accent": "#6366f1",
    "accent_hover": "#818cf8",
    "accent_glow": "#4f46e5",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "user_bubble": "#312e81",
    "user_bubble_border": "#4338ca",
    "ai_bubble": "#1e293b",
    "ai_bubble_border": "#334155",
    "gradient_start": "#6366f1",
    "gradient_end": "#a855f7",
}


class ChatBubble(ctk.CTkFrame):
    def __init__(self, master, text, is_user=False, is_system=False, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.grid_columnconfigure(0, weight=1)

        if is_system:
            self._build_system(text)
        elif is_user:
            self._build_user(text)
        else:
            self._build_ai(text)

    def _build_user(self, text):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="e", padx=(60, 0), pady=(2, 6))

        role_bar = ctk.CTkFrame(outer, fg_color="transparent")
        role_bar.pack(anchor="e", padx=14, pady=(0, 3))
        ctk.CTkLabel(role_bar, text="You", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent_hover"]).pack(side="right")

        bubble = ctk.CTkFrame(outer, fg_color=COLORS["user_bubble"],
                              corner_radius=18, border_width=1,
                              border_color=COLORS["user_bubble_border"])
        bubble.pack(anchor="e")

        self.msg = ctk.CTkLabel(bubble, text="", font=("Segoe UI", 15),
                           text_color=COLORS["text_primary"],
                           wraplength=480, justify="left", anchor="w")
        self.msg.pack(padx=18, pady=(12, 12))
        self._animate_text(text)

    def _build_ai(self, text):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="w", padx=(0, 60), pady=(2, 6))

        role_bar = ctk.CTkFrame(outer, fg_color="transparent")
        role_bar.pack(anchor="w", padx=14, pady=(0, 3))

        dot = ctk.CTkFrame(role_bar, fg_color=COLORS["success"],
                           width=8, height=8, corner_radius=4)
        dot.pack(side="left", padx=(0, 6), pady=2)
        ctk.CTkLabel(role_bar, text="GAIze AI", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["success"]).pack(side="left")

        bubble = ctk.CTkFrame(outer, fg_color=COLORS["ai_bubble"],
                              corner_radius=18, border_width=1,
                              border_color=COLORS["ai_bubble_border"])
        bubble.pack(anchor="w")

        self.msg = ctk.CTkLabel(bubble, text="", font=("Segoe UI", 15),
                           text_color=COLORS["text_primary"],
                           wraplength=480, justify="left", anchor="w")
        self.msg.pack(padx=18, pady=(12, 12))
        self._animate_text(text)

    def _build_system(self, text):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="ew", pady=(4, 4))

        pill = ctk.CTkFrame(outer, fg_color=COLORS["bg_tertiary"],
                            corner_radius=12, border_width=1,
                            border_color=COLORS["border"])
        pill.pack(anchor="center")

        self.msg = ctk.CTkLabel(pill, text="", font=("Segoe UI", 13),
                     text_color=COLORS["text_muted"])
        self.msg.pack(padx=16, pady=6)
        self._animate_text(text)

    def _animate_text(self, full_text, current_index=0):
        if current_index <= len(full_text):
            self.msg.configure(text=full_text[:current_index])

            self.after(5, self._animate_text, full_text, current_index + 1)



class StreamingBubble(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="w", padx=(0, 60), pady=(2, 6))

        role_bar = ctk.CTkFrame(outer, fg_color="transparent")
        role_bar.pack(anchor="w", padx=14, pady=(0, 3))

        self._dot = ctk.CTkFrame(role_bar, fg_color=COLORS["success"],
                                 width=8, height=8, corner_radius=4)
        self._dot.pack(side="left", padx=(0, 6), pady=2)
        ctk.CTkLabel(role_bar, text="GAIze AI", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["success"]).pack(side="left")
        self._typing_label = ctk.CTkLabel(role_bar, text="  typing",
                                          font=("Segoe UI", 12, "italic"),
                                          text_color=COLORS["text_muted"])
        self._typing_label.pack(side="left")

        self._bubble = ctk.CTkFrame(outer, fg_color=COLORS["ai_bubble"],
                                    corner_radius=18, border_width=1,
                                    border_color=COLORS["ai_bubble_border"])
        self._bubble.pack(anchor="w")

        self._msg = ctk.CTkLabel(self._bubble, text="", font=("Segoe UI", 15),
                                 text_color=COLORS["text_primary"],
                                 wraplength=480, justify="left", anchor="w")
        self._msg.pack(padx=18, pady=(12, 12))

        self._text_parts = []
        self._is_typing = True
        self._typing_step = 0
        self._animate_typing()

    def _animate_typing(self):
        if not self._is_typing:
            return
        dots = ["   ", ".  ", ".. ", "..."]
        self._typing_step = (self._typing_step + 1) % 4
        self._typing_label.configure(text=f"  typing{dots[self._typing_step]}")
        self.after(400, self._animate_typing)

    def append_token(self, token):
        self._text_parts.append(token)
        full = "".join(self._text_parts)
        self._msg.configure(text=full)

    def finalize(self):
        self._is_typing = False
        self._typing_label.configure(text="")

    def get_full_text(self):
        return "".join(self._text_parts)


class AIChatWidget(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=COLORS["bg_primary"],
                         corner_radius=16, **kwargs)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._messages = []
        self._analysis_context = ""
        self._is_generating = False
        self._streaming_bubble = None

        self._build_header()
        self._build_chat_area()
        self._build_input_bar()
        self._show_welcome()
        self.after(500, self._check_ollama)

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"],
                              corner_radius=16, border_width=1,
                              border_color=COLORS["border"])
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        header.grid_columnconfigure(1, weight=1)

        icon_ring = ctk.CTkFrame(header, fg_color=COLORS["accent"],
                                 corner_radius=20, width=40, height=40)
        icon_ring.grid(row=0, column=0, padx=(16, 10), pady=12)
        icon_ring.grid_propagate(False)

        icon_inner = ctk.CTkFrame(icon_ring, fg_color=COLORS["bg_secondary"],
                                  corner_radius=17, width=34, height=34)
        icon_inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(icon_inner, text="⚡", font=("Segoe UI Emoji", 15)).place(
            relx=0.5, rely=0.5, anchor="center")

        info_block = ctk.CTkFrame(header, fg_color="transparent")
        info_block.grid(row=0, column=1, sticky="w", pady=12)

        title_row = ctk.CTkFrame(info_block, fg_color="transparent")
        title_row.pack(anchor="w")
        ctk.CTkLabel(title_row, text="GAIze-Sport AI",
                     font=("Segoe UI", 17, "bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        self._status_dot = ctk.CTkFrame(title_row, fg_color=COLORS["text_muted"],
                                        width=8, height=8, corner_radius=4)
        self._status_dot.pack(side="left", padx=(8, 0), pady=1)

        self._status_label = ctk.CTkLabel(info_block, text="Checking connection…",
                                          font=("Segoe UI", 13),
                                          text_color=COLORS["text_muted"])
        self._status_label.pack(anchor="w")

        badge = ctk.CTkFrame(header, fg_color=COLORS["bg_tertiary"],
                             corner_radius=8, border_width=1,
                             border_color=COLORS["border"])
        badge.grid(row=0, column=2, padx=14, pady=12)
        ctk.CTkLabel(badge, text=f"⚡ {MODEL_NAME}",
                     font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["text_muted"]).pack(padx=10, pady=5)

    def _build_chat_area(self):
        chat_outer = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"],
                                  corner_radius=16, border_width=1,
                                  border_color=COLORS["border"])
        chat_outer.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        chat_outer.grid_rowconfigure(0, weight=1)
        chat_outer.grid_columnconfigure(0, weight=1)

        self._scroll = ctk.CTkScrollableFrame(
            chat_outer, fg_color=COLORS["bg_secondary"],
            corner_radius=0, scrollbar_button_color=COLORS["bg_tertiary"],
            scrollbar_button_hover_color=COLORS["surface_hover"])
        self._scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._scroll.grid_columnconfigure(0, weight=1)

        self._bubble_row = 0

    def _build_input_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"],
                           corner_radius=16, border_width=1,
                           border_color=COLORS["border"])
        bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        bar.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=10)
        inner.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            inner, font=("Segoe UI", 15), height=48,
            placeholder_text="Ask about the analysis, sport rules, AI models…",
            fg_color=COLORS["bg_primary"],
            border_color=COLORS["border"],
            corner_radius=12, border_width=1,
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"])
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", self.send_message)
        self.entry.bind("<FocusIn>", lambda e: self.entry.configure(
            border_color=COLORS["border_focus"]))
        self.entry.bind("<FocusOut>", lambda e: self.entry.configure(
            border_color=COLORS["border"]))

        self.send_btn = ctk.CTkButton(
            inner, text="↑", width=48, height=48,
            font=("Segoe UI", 20, "bold"), corner_radius=12,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#ffffff",
            command=self.send_message)
        self.send_btn.grid(row=0, column=1, padx=(0, 4))

        self._btn_clear = ctk.CTkButton(
            inner, text="⟳", width=48, height=48,
            font=("Segoe UI", 18), corner_radius=12,
            fg_color=COLORS["bg_primary"],
            hover_color=COLORS["surface_hover"],
            border_color=COLORS["border"], border_width=1,
            text_color=COLORS["text_muted"],
            command=self._clear_chat)
        self._btn_clear.grid(row=0, column=2)

    def _add_bubble(self, text, is_user=False, is_system=False):
        bubble = ChatBubble(self._scroll, text, is_user=is_user,
                            is_system=is_system)
        bubble.grid(row=self._bubble_row, column=0, sticky="ew", padx=6)
        self._bubble_row += 1
        self._scroll_to_bottom()

    def _create_streaming_bubble(self):
        sb = StreamingBubble(self._scroll)
        sb.grid(row=self._bubble_row, column=0, sticky="ew", padx=6)
        self._bubble_row += 1
        self._streaming_bubble = sb
        return sb

    def _scroll_to_bottom(self):
        self.after(60, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    def _show_welcome(self):
        self._add_bubble("🛡️ Chat session started", is_system=True)

        welcome_text = (
            "Welcome to GAIze-Sport AI Assistant!\n\n"
            "I can help you with:\n"
            "• Interpreting your video analysis results\n"
            "• Understanding action recognition data\n"
            "• Sports rules and tactical insights\n"
            "• How the AI models work\n\n"
            f"Powered by Ollama · {MODEL_NAME}"
        )
        self._add_bubble(welcome_text, is_user=False)

    def set_analysis_context(self, context_str):
        self._analysis_context = context_str

    def send_message(self, event=None):
        message = self.entry.get().strip()
        if not message or self._is_generating:
            return

        self.entry.delete(0, "end")
        self._add_bubble(message, is_user=True)

        system_content = SYSTEM_PROMPT
        if self._analysis_context:
            system_content += f"\n--- Current Analysis Context ---\n{self._analysis_context}\n"

        if not self._messages:
            self._messages.append({"role": "system", "content": system_content})

        self._messages.append({"role": "user", "content": message})

        self._is_generating = True
        self.send_btn.configure(state="disabled", fg_color=COLORS["surface"])
        self._status_label.configure(text="Generating response…",
                                     text_color=COLORS["warning"])
        self._status_dot.configure(fg_color=COLORS["warning"])

        threading.Thread(target=self._call_ollama, daemon=True).start()

    def _call_ollama(self):
        self.after(0, self._create_streaming_bubble)
        self.after(50, lambda: None)
        time.sleep(0.1)

        payload = json.dumps({
            "model": MODEL_NAME,
            "messages": self._messages,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})

        full_response = []

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_response.append(token)
                        self.after(0, lambda t=token: self._stream_token(t))

                    if chunk.get("done"):
                        break

        except urllib.error.URLError:
            self.after(0, lambda: self._stream_token(
                "\n⚠ Could not connect to Ollama.\n"
                "Make sure it is running: ollama serve"))
        except Exception as e:
            self.after(0, lambda: self._stream_token(f"\n⚠ Error: {e}"))

        complete_text = "".join(full_response)
        if complete_text:
            self._messages.append({"role": "assistant", "content": complete_text})

        self.after(0, self._finalize_stream)

    def _stream_token(self, token):
        if self._streaming_bubble:
            self._streaming_bubble.append_token(token)
            self._scroll_to_bottom()

    def _finalize_stream(self):
        if self._streaming_bubble:
            self._streaming_bubble.finalize()
            self._streaming_bubble = None
        self._is_generating = False
        self.send_btn.configure(state="normal", fg_color=COLORS["accent"])
        self._status_label.configure(text="Ready", text_color=COLORS["success"])
        self._status_dot.configure(fg_color=COLORS["success"])
        self._scroll_to_bottom()

    def _clear_chat(self):
        if self._is_generating:
            return
        self._messages.clear()

        for widget in self._scroll.winfo_children():
            widget.destroy()
        self._bubble_row = 0

        self._show_welcome()
        self._status_label.configure(text="Chat cleared",
                                     text_color=COLORS["text_muted"])
        self._status_dot.configure(fg_color=COLORS["text_muted"])
        self.after(2000, lambda: self._status_label.configure(
            text="Ready", text_color=COLORS["success"]))
        self.after(2000, lambda: self._status_dot.configure(
            fg_color=COLORS["success"]))

    def _check_ollama(self):
        threading.Thread(target=self._ping_ollama, daemon=True).start()

    def _ping_ollama(self):
        try:
            req = urllib.request.Request(f"http://localhost:{OLLAMA_PORT}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                has_model = any(MODEL_NAME in m for m in models)
                if has_model:
                    self.after(0, lambda: (
                        self._status_label.configure(
                            text=f"Connected · {MODEL_NAME} ready",
                            text_color=COLORS["success"]),
                        self._status_dot.configure(fg_color=COLORS["success"])
                    ))
                else:
                    self.after(0, lambda: (
                        self._status_label.configure(
                            text=f"{MODEL_NAME} not found — run: ollama pull {MODEL_NAME}",
                            text_color=COLORS["warning"]),
                        self._status_dot.configure(fg_color=COLORS["warning"])
                    ))
        except Exception:
            self.after(0, lambda: (
                self._status_label.configure(
                    text="Ollama not running — run: ollama serve",
                    text_color=COLORS["error"]),
                self._status_dot.configure(fg_color=COLORS["error"])
            ))
