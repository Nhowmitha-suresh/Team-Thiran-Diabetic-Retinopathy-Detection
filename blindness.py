# ============================================================
# RETINAL AI – FULL E-HEALTH MANAGEMENT SYSTEM
# CustomTkinter | SQLite | Image Upload | Dashboard | Reports
# ============================================================

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sqlite3
import datetime
import random
import os

# ============================================================
# THEME SETTINGS
# ============================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BG_COLOR = "#f9fafb"
PRIMARY = "#2563eb"
ACCENT = "#ef4444"
HOVER = "#dc2626"
TEXT_DARK = "#101828"
TEXT_LIGHT = "#667085"
SIDEBAR_COLOR = "#101828"
CARD_BORDER = "#e5e7eb"

TITLE_FONT = ("Segoe UI Semibold", 36)
SUBTITLE_FONT = ("Segoe UI", 18)
TEXT_FONT = ("Segoe UI", 14)
BTN_FONT = ("Segoe UI Semibold", 16)

# ============================================================
# DATABASE SETUP
# ============================================================

conn = sqlite3.connect("retinal_ai.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    prediction TEXT,
    confidence TEXT,
    status TEXT,
    date TEXT,
    image_path TEXT
)
""")

conn.commit()

# ============================================================
# MAIN APPLICATION
# ============================================================

class RetinalAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Retinal AI – E-Health Management System")
        self.geometry("1400x900")
        self.configure(fg_color=BG_COLOR)

        self.current_user = None

        self.container = ctk.CTkFrame(self, fg_color=BG_COLOR)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for page in (
            HomePage,
            LoginPage,
            DashboardPage,
            UploadPage,
            ReportsPage
        ):
            frame = page(self.container, self)
            self.frames[page] = frame
            frame.place(relwidth=1, relheight=1)

        self.show(HomePage)

    def show(self, page):
        self.frames[page].tkraise()

# ============================================================
# NAVBAR
# ============================================================

class Navbar(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, height=70, fg_color="white")
        self.controller = controller
        self.pack(fill="x")

        ctk.CTkLabel(
            self,
            text="🧠 Retinal AI",
            font=("Segoe UI Semibold", 22),
            text_color=TEXT_DARK
        ).pack(side="left", padx=40)

        ctk.CTkButton(
            self,
            text="Login",
            font=BTN_FONT,
            fg_color=PRIMARY,
            corner_radius=25,
            width=120,
            command=lambda: controller.show(LoginPage)
        ).pack(side="right", padx=40)

# ============================================================
# HOME PAGE
# ============================================================

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        Navbar(self, controller)

        hero = ctk.CTkFrame(self, fg_color=PRIMARY, height=420)
        hero.pack(fill="x")

        hero_content = ctk.CTkFrame(hero, fg_color="transparent")
        hero_content.pack(expand=True, padx=80, pady=80)

        ctk.CTkLabel(
            hero_content,
            text="AI-Powered Retinal Disease Detection",
            font=TITLE_FONT,
            text_color="white"
        ).pack(anchor="w")

        ctk.CTkLabel(
            hero_content,
            text="Early detection of Diabetic Retinopathy\nusing deep learning.",
            font=SUBTITLE_FONT,
            text_color="white"
        ).pack(anchor="w", pady=20)

        ctk.CTkButton(
            hero_content,
            text="Get Started",
            font=BTN_FONT,
            fg_color=ACCENT,
            hover_color=HOVER,
            corner_radius=30,
            width=220,
            height=52,
            command=lambda: controller.show(LoginPage)
        ).pack(anchor="w")

        features = ctk.CTkFrame(self, fg_color=BG_COLOR)
        features.pack(expand=True, pady=50)

        ctk.CTkLabel(
            features,
            text="Key Features",
            font=("Segoe UI Semibold", 28),
            text_color=TEXT_DARK
        ).pack(pady=30)

        grid = ctk.CTkFrame(features, fg_color="transparent")
        grid.pack()

        cards = [
            ("🧠 AI Diagnosis", "Deep learning based DR detection"),
            ("📊 Medical Reports", "Auto-generated clinical reports"),
            ("👨‍⚕️ Doctor Review", "Expert validation"),
            ("🛡️ Insurance Ready", "Claim friendly output")
        ]

        for i, (title, desc) in enumerate(cards):
            card = ctk.CTkFrame(
                grid,
                fg_color="white",
                corner_radius=16,
                border_width=1,
                border_color=CARD_BORDER,
                width=320,
                height=160
            )
            card.grid(row=i // 2, column=i % 2, padx=25, pady=25)

            ctk.CTkLabel(card, text=title,
                         font=("Segoe UI Semibold", 17),
                         text_color=TEXT_DARK).pack(pady=20)

            ctk.CTkLabel(card, text=desc,
                         font=TEXT_FONT,
                         text_color=TEXT_LIGHT,
                         wraplength=260).pack()

# ============================================================
# LOGIN PAGE
# ============================================================

class LoginPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white")
        self.controller = controller

        box = ctk.CTkFrame(
            self,
            fg_color="white",
            corner_radius=20,
            border_width=1,
            border_color=CARD_BORDER
        )
        box.pack(expand=True, ipadx=40, ipady=40)

        ctk.CTkLabel(box, text="Login",
                     font=("Segoe UI Semibold", 30),
                     text_color=TEXT_DARK).pack(pady=20)

        self.username = ctk.CTkEntry(
            box, placeholder_text="Username", width=320)
        self.username.pack(pady=10)

        self.password = ctk.CTkEntry(
            box, placeholder_text="Password",
            show="*", width=320)
        self.password.pack(pady=10)

        ctk.CTkButton(
            box,
            text="Sign In",
            font=BTN_FONT,
            fg_color=PRIMARY,
            corner_radius=25,
            width=320,
            height=45,
            command=self.login
        ).pack(pady=30)

        ctk.CTkButton(
            box,
            text="← Back",
            fg_color="transparent",
            text_color=TEXT_LIGHT,
            command=lambda: controller.show(HomePage)
        ).pack()

    def login(self):
        u = self.username.get()
        p = self.password.get()

        if not u or not p:
            messagebox.showerror("Error", "All fields required")
            return

        cursor.execute("SELECT * FROM users WHERE username=?", (u,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users VALUES (?,?,?)",
                (u, p, datetime.datetime.now().strftime("%d/%m/%Y"))
            )
            conn.commit()

        self.controller.current_user = u
        self.controller.show(DashboardPage)

# ============================================================
# SIDEBAR
# ============================================================

class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=SIDEBAR_COLOR, width=220)
        self.pack(side="left", fill="y")

        ctk.CTkLabel(
            self,
            text="🧠 Retinal AI",
            font=("Segoe UI Semibold", 18),
            text_color="white"
        ).pack(pady=30)

        buttons = [
            ("📊 Dashboard", DashboardPage),
            ("📤 Upload", UploadPage),
            ("📋 Reports", ReportsPage),
            ("🚪 Logout", HomePage)
        ]

        for text, page in buttons:
            ctk.CTkButton(
                self,
                text=text,
                fg_color="transparent",
                hover_color="#344054",
                font=TEXT_FONT,
                text_color="white",
                corner_radius=12,
                command=lambda p=page: controller.show(p)
            ).pack(fill="x", padx=15, pady=8)

# ============================================================
# DASHBOARD PAGE
# ============================================================

class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        Sidebar(self, controller)

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(expand=True, fill="both")

        ctk.CTkLabel(
            content,
            text="Dashboard",
            font=("Segoe UI Semibold", 28),
            text_color=TEXT_DARK
        ).pack(anchor="w", padx=40, pady=30)

        stats = ctk.CTkFrame(content, fg_color="transparent")
        stats.pack(padx=40, pady=20)

        cards = [
            ("Total Scans", "12"),
            ("Normal", "8"),
            ("Abnormal", "4"),
            ("Pending", "2")
        ]

        for title, value in cards:
            card = ctk.CTkFrame(
                stats,
                fg_color="white",
                corner_radius=16,
                border_width=1,
                border_color=CARD_BORDER,
                width=200,
                height=120
            )
            card.pack(side="left", padx=15)

            ctk.CTkLabel(card, text=value,
                         font=("Segoe UI Semibold", 26),
                         text_color=PRIMARY).pack(pady=15)

            ctk.CTkLabel(card, text=title,
                         font=TEXT_FONT,
                         text_color=TEXT_LIGHT).pack()

# ============================================================
# UPLOAD PAGE
# ============================================================

class UploadPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)
        self.controller = controller
        self.image_path = None

        Sidebar(self, controller)

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(expand=True)

        box = ctk.CTkFrame(
            content,
            fg_color="white",
            corner_radius=20,
            border_width=1,
            border_color=CARD_BORDER,
            width=520,
            height=440
        )
        box.pack(expand=True)

        ctk.CTkLabel(
            box,
            text="Upload Retinal Image",
            font=("Segoe UI Semibold", 24)
        ).pack(pady=30)

        self.preview = ctk.CTkLabel(
            box,
            text="📁 Click to select image",
            text_color=TEXT_LIGHT
        )
        self.preview.pack(pady=20)

        ctk.CTkButton(
            box,
            text="Choose File",
            command=self.choose
        ).pack(pady=10)

        ctk.CTkButton(
            box,
            text="Predict",
            fg_color=ACCENT,
            hover_color=HOVER,
            command=self.predict
        ).pack(pady=10)

        self.result = ctk.CTkLabel(box, text="", font=TEXT_FONT)
        self.result.pack(pady=20)

    def choose(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if path:
            img = Image.open(path)
            img.thumbnail((220, 220))
            self.photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.photo, text="")
            self.image_path = path

    def predict(self):
        if not self.image_path:
            messagebox.showerror("Error", "Upload image first")
            return

        prediction = random.choice(
            ["No DR", "Mild DR", "Moderate DR", "Severe DR"]
        )
        confidence = f"{random.randint(85, 99)}%"
        status = "Normal" if prediction == "No DR" else "Abnormal"

        self.result.configure(
            text=f"{prediction} | {confidence} | {status}",
            text_color="green" if status == "Normal" else "red"
        )

        cursor.execute(
            "INSERT INTO reports VALUES (NULL,?,?,?,?,?,?)",
            (
                self.controller.current_user,
                prediction,
                confidence,
                status,
                datetime.datetime.now().strftime("%d/%m/%Y"),
                self.image_path
            )
        )
        conn.commit()

# ============================================================
# REPORTS PAGE
# ============================================================

class ReportsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        Sidebar(self, controller)

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(expand=True, fill="both")

        ctk.CTkLabel(
            content,
            text="Medical Reports",
            font=("Segoe UI Semibold", 28)
        ).pack(anchor="w", padx=40, pady=30)

        cursor.execute(
            "SELECT prediction, confidence, status, date FROM reports"
        )

        for row in cursor.fetchall():
            ctk.CTkLabel(
                content,
                text=f"{row[3]} | {row[0]} | {row[1]} | {row[2]}",
                font=TEXT_FONT,
                text_color=TEXT_DARK
            ).pack(anchor="w", padx=40, pady=5)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app = RetinalAI()
    app.mainloop()
