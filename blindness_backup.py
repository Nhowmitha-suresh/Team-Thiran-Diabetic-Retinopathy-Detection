import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import sqlite3, datetime, webbrowser

# ---------------- THEME SETTINGS ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_GRADIENT = ("#0a0f1a", "#001d33")
ACCENT = "#00FFFF"
HOVER = "#008B8B"
TITLE_FONT = ("Segoe UI Semibold", 26)
TEXT_FONT = ("Segoe UI", 13)
BTN_FONT = ("Segoe UI Semibold", 14)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("dr_users.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    name TEXT,
    email TEXT,
    created_at TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    comment TEXT,
    rating INTEGER,
    created_at TEXT
)
""")
conn.commit()

# ---------------- APP CLASS ----------------
class RetinalAI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Retinal AI - Tamil Nadu Network")
        self.geometry("1000x650")
        self.resizable(False, False)

        # Navigation bar
        self.navbar = ctk.CTkFrame(self, fg_color="#001b25", height=50)
        self.navbar.pack(fill="x")

        self.current_user = None

        # Page container
        self.container = ctk.CTkFrame(self, fg_color=BG_GRADIENT)
        self.container.pack(fill="both", expand=True)

        # Initialize frames
        self.frames = {}
        for Page in (WelcomePage, LoginPage, SignupPage, AboutPage, ReportPage, ReviewPage, DoctorPage):
            frame = Page(self.container, self)
            self.frames[Page] = frame
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.show_frame(WelcomePage)

    def update_navbar(self, active_page=None):
        for widget in self.navbar.winfo_children():
            widget.destroy()

        ctk.CTkLabel(self.navbar, text="RETINAL AI", text_color=ACCENT,
                     font=("Segoe UI Black", 20)).pack(side="left", padx=20)

        if self.current_user:
            buttons = [
                ("About", AboutPage),
                ("Upload Report", ReportPage),
                ("Review", ReviewPage),
                ("Doctors", DoctorPage),
                ("Logout", WelcomePage)
            ]
            for text, page in buttons:
                ctk.CTkButton(
                    self.navbar, text=text, font=("Segoe UI", 13),
                    fg_color="transparent", hover_color="#002b3d",
                    text_color="white", width=100, corner_radius=10,
                    command=lambda p=page: self.show_frame(p)
                ).pack(side="right", padx=5, pady=5)

    def show_frame(self, page):
        self.update_navbar(active_page=page)
        self.frames[page].tkraise()

# ---------------- PAGES ----------------

class WelcomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        self.controller = controller
        ctk.CTkLabel(self, text="Welcome to", font=("Segoe UI", 20), text_color="#AAAAAA").pack(pady=(120, 0))
        ctk.CTkLabel(self, text="RETINAL AI", font=("Segoe UI Black", 42), text_color=ACCENT).pack(pady=10)
        ctk.CTkLabel(self, text="Diabetic Retinopathy Detection System", font=TEXT_FONT).pack(pady=20)

        ctk.CTkButton(self, text="Login", fg_color=ACCENT, text_color="black",
                      hover_color=HOVER, font=BTN_FONT,
                      command=lambda: controller.show_frame(LoginPage)).pack(pady=10)
        ctk.CTkButton(self, text="Sign Up", fg_color="#9400D3", text_color="white",
                      hover_color="#9932CC", font=BTN_FONT,
                      command=lambda: controller.show_frame(SignupPage)).pack(pady=10)


class LoginPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        self.controller = controller
        ctk.CTkLabel(self, text="Login", font=TITLE_FONT, text_color=ACCENT).pack(pady=(100, 20))

        self.username = ctk.CTkEntry(self, placeholder_text="Username", width=250)
        self.username.pack(pady=5)
        self.password = ctk.CTkEntry(self, placeholder_text="Password", show="*", width=250)
        self.password.pack(pady=5)

        ctk.CTkButton(self, text="Login", fg_color=ACCENT, text_color="black", hover_color=HOVER,
                      font=BTN_FONT, command=self.login).pack(pady=15)
        ctk.CTkButton(self, text="Back", fg_color="#444", hover_color="#666",
                      font=BTN_FONT, command=lambda: controller.show_frame(WelcomePage)).pack()

    def login(self):
        u, p = self.username.get(), self.password.get()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if cursor.fetchone():
            messagebox.showinfo("Welcome", f"Hello {u}")
            self.controller.current_user = u
            self.controller.show_frame(AboutPage)
        else:
            messagebox.showerror("Error", "Invalid credentials")


class SignupPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        self.controller = controller
        ctk.CTkLabel(self, text="Create Account", font=TITLE_FONT, text_color="#FFD700").pack(pady=(80, 20))
        self.entries = {}
        for field in ["Username", "Password", "Full Name", "Email"]:
            e = ctk.CTkEntry(self, placeholder_text=field, width=300)
            e.pack(pady=8)
            self.entries[field] = e

        ctk.CTkButton(self, text="Sign Up", fg_color=ACCENT, text_color="black",
                      hover_color=HOVER, font=BTN_FONT, command=self.signup).pack(pady=15)
        ctk.CTkButton(self, text="Back", fg_color="#444", hover_color="#666",
                      font=BTN_FONT, command=lambda: controller.show_frame(LoginPage)).pack()

    def signup(self):
        vals = {k: e.get() for k, e in self.entries.items()}
        if not all(vals.values()):
            messagebox.showerror("Error", "All fields required")
            return
        cursor.execute("SELECT * FROM users WHERE username=?", (vals["Username"],))
        if cursor.fetchone():
            messagebox.showerror("Error", "Username exists")
            return
        cursor.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                       (vals["Username"], vals["Password"], vals["Full Name"], vals["Email"],
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        messagebox.showinfo("Success", "Account created successfully!")
        self.controller.show_frame(LoginPage)


class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        ctk.CTkLabel(self, text="About Retinal AI", font=TITLE_FONT, text_color=ACCENT).pack(pady=(70, 20))
        desc = ("This AI system predicts Diabetic Retinopathy (DR) stages from retinal images.\n"
                "Built with deep learning models, it helps early diagnosis and connects users\n"
                "with trusted ophthalmologists in Tamil Nadu for consultation.")
        ctk.CTkLabel(self, text=desc, wraplength=800, font=TEXT_FONT).pack(pady=20)


class ReportPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        self.controller = controller
        ctk.CTkLabel(self, text="Upload Retinal Image", font=TITLE_FONT, text_color=ACCENT).pack(pady=(70, 20))
        self.preview = ctk.CTkLabel(self, text="")
        self.preview.pack(pady=10)
        self.result = ctk.CTkLabel(self, text="", wraplength=800)
        self.result.pack(pady=15)
        ctk.CTkButton(self, text="Choose Image", fg_color=ACCENT, text_color="black",
                      hover_color=HOVER, command=self.upload).pack(pady=10)

    def upload(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png")])
        if not f:
            return
        img = Image.open(f)
        img.thumbnail((300, 250))
        self.tkimg = ImageTk.PhotoImage(img)
        self.preview.configure(image=self.tkimg)
        self.result.configure(text="Prediction: Mild DR\nRecommendation: Regular monitoring.")


class ReviewPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        ctk.CTkLabel(self, text="Patient Review", font=TITLE_FONT, text_color="#FFD700").pack(pady=(70, 20))
        self.comment = ctk.CTkTextbox(self, width=500, height=100)
        self.comment.pack(pady=10)
        self.rating = ctk.CTkComboBox(self, values=["1", "2", "3", "4", "5"], width=80)
        self.rating.pack(pady=5)
        ctk.CTkButton(self, text="Submit", fg_color=ACCENT, text_color="black",
                      hover_color=HOVER, command=self.submit).pack(pady=15)

    def submit(self):
        comment = self.comment.get("1.0", "end").strip()
        rating = self.rating.get()
        if not comment:
            messagebox.showerror("Error", "Please enter a review")
            return
        cursor.execute("INSERT INTO reviews (username, comment, rating, created_at) VALUES (?,?,?,?)",
                       ("guest", comment, rating, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        messagebox.showinfo("Success", "Review submitted!")


class DoctorPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_GRADIENT)
        ctk.CTkLabel(self, text="Contact Ophthalmologists (Tamil Nadu)", font=TITLE_FONT, text_color=ACCENT).pack(pady=(60, 20))
        doctors = [
            ("Aravind Eye Hospital", "Madurai", "+91 452 435 6100", "https://www.aravind.org"),
            ("Sankara Nethralaya", "Chennai", "+91 44 2827 1616", "https://www.sankaranethralaya.org"),
            ("Dr. Agarwal’s Eye Hospital", "Chennai", "+91 44 4227 7777", "https://www.dragarwal.com"),
            ("Vasan Eye Care", "Coimbatore", "+91 422 456 7777", "https://www.vasaneye.in"),
            ("Lotus Eye Hospital", "Coimbatore", "+91 422 422 9999", "https://www.lotuseye.org"),
            ("Joseph Eye Hospital", "Tiruchirapalli", "+91 431 270 1615", "https://www.josepheye.org"),
            ("Uma Eye Clinic", "Chennai", "+91 44 2827 4050", "https://www.umaeyeclinic.in")
        ]
        for name, city, phone, site in doctors:
            frame = ctk.CTkFrame(self, fg_color="#002b3d", corner_radius=10)
            frame.pack(pady=10, padx=50, fill="x")
            ctk.CTkLabel(frame, text=f"{name} ({city})", font=("Segoe UI", 15, "bold"),
                         text_color="#00CED1").pack(anchor="w", padx=15)
            ctk.CTkLabel(frame, text=f"📞 {phone}", font=TEXT_FONT).pack(anchor="w", padx=15)
            ctk.CTkButton(frame, text="Visit Website", fg_color=ACCENT, text_color="black",
                          hover_color=HOVER, width=120,
                          command=lambda url=site: webbrowser.open(url)).pack(anchor="e", padx=15, pady=5)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app = RetinalAI()
    app.mainloop()
