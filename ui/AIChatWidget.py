import customtkinter as ctk

class AIChatWidget(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.chat_history = ctk.CTkTextbox(self, state="disabled")
        self.chat_history.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask something...")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", width=60, command=self.send_message)
        self.send_btn.grid(row=0, column=1)

    def send_message(self, event=None):
        message = self.entry.get()
        if message:
            self.display_message("You", message)
            self.entry.delete(0, "end")
            self.after(500, lambda: self.display_message("AI", "This is a mock response."))

    def display_message(self, sender, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"{sender}: {message}\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")
