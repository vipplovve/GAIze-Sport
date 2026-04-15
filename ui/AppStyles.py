import customtkinter as ctk

THEME_COLOR = "dark-blue"
APPEARANCE_MODE = "Dark"

COLOR_PRIMARY = "#1f6aa5"
COLOR_SECONDARY = "#144870"
COLOR_ACCENT = "#2cc985"
COLOR_TEXT = "#ffffff"
COLOR_TEXT_SECONDARY = "#a0a0a0"
COLOR_BG = "#1a1a1a"
COLOR_PANEL_BG = "#2b2b2b"

FONT_FAMILY = "Roboto"
FONT_SIZE_TITLE = 24
FONT_SIZE_HEADER = 18
FONT_SIZE_NORMAL = 14
FONT_SIZE_SMALL = 12

PADDING = 20
CORNER_RADIUS = 10
BUTTON_HEIGHT = 40

def setup_theme():
    ctk.set_appearance_mode(APPEARANCE_MODE)
    ctk.set_default_color_theme(THEME_COLOR)

def get_title_font():
    return (FONT_FAMILY, FONT_SIZE_TITLE, "bold")

def get_header_font():
    return (FONT_FAMILY, FONT_SIZE_HEADER, "bold")

def get_normal_font():
    return (FONT_FAMILY, FONT_SIZE_NORMAL)
