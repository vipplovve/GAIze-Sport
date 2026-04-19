import customtkinter as ctk
from ui.AppStyles import setup_theme
from ui.LoginScreen import LoginScreen
from ui.MainDashboard import MainDashboard

class SportsAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GAIze-Sport — AI Sports Video Analysis")
        self.geometry("1100x700")
        setup_theme()
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.current_screen = None
        self.show_login()

    def show_login(self):
        if self.current_screen:
            self.current_screen.destroy()
        self.login_screen = LoginScreen(master=self, on_login_success=self.on_login)
        self.login_screen.grid(row=0, column=0, sticky="nsew")
        self.current_screen = self.login_screen

    def on_login(self, username):
        if self.current_screen:
            self.current_screen.destroy()
        self.dashboard = MainDashboard(master=self, on_logout=self.show_login)
        self.dashboard.grid(row=0, column=0, sticky="nsew")
        self.current_screen = self.dashboard

if __name__ == "__main__":
    app = SportsAnalysisApp()
    app.mainloop()
