
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab


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

        # ---- Canvas Area ----
        self.canvas = tk.Canvas(self.root, bg="lightgray")
        self.canvas.pack(fill="both", expand=True)

        # Status bar
        self.status = tk.Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # ---- Toolbar ----
        toolbar = tk.Frame(self.root, bg="#ececec", height=40)
        toolbar.pack(side="top", fill="x")
        tk.Button(toolbar, text="🖌️ Draw", command=self.enable_draw).pack(side="left", padx=5)
        tk.Button(toolbar, text="😀 Emoji", command=self.add_emoji_mode).pack(side="left", padx=5)
        tk.Button(toolbar, text="↩️ Undo", command=self.undo_action).pack(side="left", padx=5)
        tk.Button(toolbar, text="↪️ Redo", command=self.redo_action).pack(side="left", padx=5)
        tk.Button(toolbar, text="❌ Clear", command=self.clear_canvas).pack(side="left", padx=5)
        tk.Button(toolbar, text="💾 Save", command=self.save_image).pack(side="left", padx=5)

        # ---- Undo/Redo Stacks ----
        self.undo_stack = []
        self.redo_stack = []

        # ---- Control Panel ----
        control_panel = tk.Frame(self.root, bg="#f5f5f5", width=120)
        control_panel.pack(side="left", fill="y")
        tk.Label(control_panel, text="Brush Size:").pack(pady=(10,0))
        self.brush_size_slider = tk.Scale(control_panel, from_=1, to=20, orient="horizontal")
        self.brush_size_slider.set(self.pen_size)
        self.brush_size_slider.pack(pady=(0,10))
        tk.Label(control_panel, text="Brush Color:").pack()
        self.color_btn = tk.Button(control_panel, text="Choose Color", command=self.choose_color)
        self.color_btn.pack(pady=(0,10))
    def undo_action(self):
        if self.undo_stack:
            item = self.undo_stack.pop()
            self.redo_stack.append(item)
            self.canvas.delete(item['id'])
        else:
            pass

    def redo_action(self):
        if self.redo_stack:
            item = self.redo_stack.pop()
            if item.get('type') == 'emoji':
                obj = self.canvas.create_text(item['x'], item['y'], text=item['emoji'], font=("Arial", item['size']))
                item['id'] = obj
                self.undo_stack.append(item)
            elif item.get('type') == 'line':
                obj = self.canvas.create_line(item['coords'], fill=item['color'], width=item['width'], capstyle="round")
                item['id'] = obj
                self.undo_stack.append(item)
            pass
        else:
            pass
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

    def add_emoji_mode(self):
        # Show emoji selection dialog with both scrollbars and text entry for size
        emojis = [
            "😀", "😂", "😍", "😎", "🥳", "😜", "🤩", "😭", "😡", "👍", "👀", "🎉", "💖", "🔥", "🤖",
            "😇", "😏", "😱", "😴", "😈", "👻", "💩", "🙈", "🙉", "🙊", "🐶", "🐱", "🦄", "🍕", "🍔", "🍟",
            "🍦", "🍩", "🍉", "🍓", "🍒", "🍇", "🍌", "🍍", "🥑", "🥦", "🥕", "🌈", "⭐", "⚡", "☀️", "🌙"
        ]
        self.selected_emoji = None
        self.emoji_size = 32
        def select_emoji(e):
            self.selected_emoji = e
            self.emoji_size = size_scale.get()
            emoji_win.destroy()

        emoji_win = tk.Toplevel(self.root)
        emoji_win.title("Choose an Emoji")
        emoji_win.geometry("400x300")

        size_label = tk.Label(emoji_win, text="Emoji Size:")
        size_label.pack(pady=(10,0))
        size_scale = tk.Scale(emoji_win, from_=16, to=128, orient="horizontal")
        size_scale.set(self.emoji_size)
        size_scale.pack(pady=(0,10))

        canvas = tk.Canvas(emoji_win, width=350, height=200)
        frame = tk.Frame(canvas)
        vsb = tk.Scrollbar(emoji_win, orient="vertical", command=canvas.yview)
        hsb = tk.Scrollbar(emoji_win, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.create_window((0,0), window=frame, anchor="nw")

        # Grid: 8 columns for horizontal scroll
        for i, emoji in enumerate(emojis):
            btn = tk.Button(frame, text=emoji, font=("Arial", 24), command=lambda e=emoji: select_emoji(e))
            btn.grid(row=i//8, column=i%8, padx=5, pady=5)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", on_frame_configure)

        emoji_win.grab_set()
        self.root.wait_window(emoji_win)
        if self.selected_emoji:
            self.canvas.bind("<Button-1>", self.place_emoji)

    def place_emoji(self, event):
        x, y = event.x, event.y
        obj = self.canvas.create_text(x, y, text=self.selected_emoji, font=("Arial", self.emoji_size))
        self.undo_stack.append({'type': 'emoji', 'x': x, 'y': y, 'emoji': self.selected_emoji, 'size': self.emoji_size, 'id': obj})
        self.canvas.unbind("<Button-1>")

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
            # Add image resize option with draggable scale
            resize_win = tk.Toplevel(self.root)
            resize_win.title("Resize Image")
            w, h = self.image.size
            tk.Label(resize_win, text="Width:").pack()
            width_scale = tk.Scale(resize_win, from_=int(w/4), to=int(w*2), orient="horizontal")
            width_scale.set(w)
            width_scale.pack()
            tk.Label(resize_win, text="Height:").pack()
            height_scale = tk.Scale(resize_win, from_=int(h/4), to=int(h*2), orient="horizontal")
            height_scale.set(h)
            height_scale.pack()
            def apply_resize():
                try:
                    new_w = width_scale.get()
                    new_h = height_scale.get()
                    self.image = self.image.resize((new_w, new_h), Image.ANTIALIAS)
                except Exception:
                    pass
                resize_win.destroy()
            tk.Button(resize_win, text="Resize", command=apply_resize).pack(pady=10)
            resize_win.grab_set()
            self.root.wait_window(resize_win)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save_image(self):
        # Save the canvas content (including drawings and emojis) using pyscreenshot
        self.root.update()
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if path:
            try:
                import pyscreenshot as ImageGrab
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                img.save(path)
                self.status.config(text=f"Saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

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
            obj = self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill=self.pen_color, width=self.pen_size, capstyle="round")
            self.undo_stack.append({'type': 'line', 'coords': (self.last_x, self.last_y, x, y), 'color': self.pen_color, 'width': self.pen_size, 'id': obj})
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