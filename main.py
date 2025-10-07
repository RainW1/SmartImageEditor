import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from pillow import Image, ImageTk

root = tk.Tk()
root.title("Local Image Loader")
img = Image.open("asset")
label = tk.Label(root, image=photo)
label.pack()



class SmartImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Image Editor")
        self.root.geometry("1000x700")

        # ---- Variables ----
        self.image = None
        self.tk_image = None
        self.file_path = None
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.pen_color = "black"
        self.pen_size = 3

        # ---- Menu Bar ----
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save As", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Choose Color", command=self.choose_color)
        edit_menu.add_command(label="Clear Canvas", command=self.clear_canvas)

        # ---- Toolbar ----
        toolbar = tk.Frame(self.root, bg="#ececec", height=40)
        toolbar.pack(side="top", fill="x")

        tk.Button(toolbar, text="🖌️ Draw", command=self.enable_draw).pack(side="left", padx=5)
        tk.Button(toolbar, text="❌ Clear", command=self.clear_canvas).pack(side="left", padx=5)
        tk.Button(toolbar, text="🎨 Color", command=self.choose_color).pack(side="left", padx=5)
        tk.Button(toolbar, text="💾 Save", command=self.save_image).pack(side="left", padx=5)

        # ---- Canvas Area ----
        self.canvas = tk.Canvas(self.root, bg="lightgray")
        self.canvas.pack(fill="both", expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Status bar
        self.status = tk.Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    # ---- Functions ----
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.file_path = path
            self.image = Image.open(path)
            self.display_image()
            self.status.config(text=f"Opened: {path}")

    def display_image(self):
        if self.image:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save_image(self):
        if not self.image:
            messagebox.showerror("Error", "No image to save!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if path:
            self.image.save(path)
            self.status.config(text=f"Saved: {path}")

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose drawing color")
        if color[1]:
            self.pen_color = color[1]
            self.status.config(text=f"Selected Color: {self.pen_color}")

    def enable_draw(self):
        self.status.config(text="Drawing mode enabled")

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill=self.pen_color, width=self.pen_size, capstyle="round")
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.status.config(text="Canvas cleared")

# ---- Run the App ----
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartImageEditor(root)
    root.mainloop()
root.mainloop()