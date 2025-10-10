import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
import cv2
import numpy as np
import sys
import os

# Add parent directory to path untuk import features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import LinearFilters, NonLinearFilters, EdgeDetection, GeometricTransforms
from features.color_enhancement import *


class SmartImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Image Editor")
        self.root.geometry("1000x700")

        # ---- Variables ----
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.file_path = None
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.pen_color = "black"
        self.pen_size = 3

        # ---- Menu Bar ----
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # File Menu
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save As", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit Menu
        edit_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset to Original", command=self.reset_to_original)
        edit_menu.add_separator()
        edit_menu.add_command(label="Choose Color", command=self.choose_color)
        edit_menu.add_command(label="Clear Canvas", command=self.clear_canvas)

        # Filters Menu
        filter_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Filters", menu=filter_menu)
        
        # Linear Filters submenu
        linear_menu = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Linear Filters", menu=linear_menu)
        linear_menu.add_command(label="Mean Filter", command=self.apply_mean_filter)
        linear_menu.add_command(label="Gaussian Blur", command=self.apply_gaussian_filter)
        linear_menu.add_command(label="Sharpen", command=self.apply_sharpen_filter)
        
        # Non-linear filter
        filter_menu.add_command(label="Median Filter", command=self.apply_median_filter)
        
        # Edge Detection submenu
        edge_menu = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Edge Detection", menu=edge_menu)
        edge_menu.add_command(label="Sobel Edge", command=self.apply_sobel)
        edge_menu.add_command(label="Prewitt Edge", command=self.apply_prewitt)
        edge_menu.add_command(label="Laplacian Edge", command=self.apply_laplacian)

        # Transform Menu
        transform_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Transform", menu=transform_menu)
        transform_menu.add_command(label="Crop...", command=self.crop_image_dialog)
        transform_menu.add_command(label="Resize...", command=self.resize_image_dialog)
        transform_menu.add_separator()
        transform_menu.add_command(label="Rotate 90° CW", command=lambda: self.rotate_90_degrees(3))
        transform_menu.add_command(label="Rotate 90° CCW", command=lambda: self.rotate_90_degrees(1))
        transform_menu.add_command(label="Rotate 180°", command=lambda: self.rotate_90_degrees(2))
        transform_menu.add_command(label="Rotate Custom...", command=self.rotate_custom_dialog)
        transform_menu.add_separator()
        transform_menu.add_command(label="Flip Horizontal", command=lambda: self.flip_image(1))
        transform_menu.add_command(label="Flip Vertical", command=lambda: self.flip_image(0))

        # Color Enhancement Menu
        color_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Color", menu=color_menu)
        color_menu.add_command(label="Brightness & Contrast...", command=self.adjust_brightness_contrast_dialog)
        color_menu.add_command(label="Saturation & Hue...", command=self.adjust_saturation_hue_dialog)
        color_menu.add_command(label="Gamma Correction...", command=self.adjust_gamma_dialog)
        color_menu.add_separator()
        color_menu.add_command(label="Histogram Equalization", command=self.apply_histogram_eq)
        color_menu.add_separator()
        threshold_menu = tk.Menu(color_menu, tearoff=0)
        color_menu.add_cascade(label="Thresholding", menu=threshold_menu)
        threshold_menu.add_command(label="Global Threshold...", command=self.apply_global_threshold_dialog)
        threshold_menu.add_command(label="Adaptive Threshold", command=self.apply_adaptive_threshold_fn)
        blur_menu = tk.Menu(color_menu, tearoff=0)
        color_menu.add_cascade(label="Blur Effects", menu=blur_menu)
        blur_menu.add_command(label="Average Blur...", command=self.apply_average_blur_dialog)
        blur_menu.add_command(label="Gaussian Blur...", command=self.apply_gaussian_blur_dialog)
        blur_menu.add_command(label="Median Blur...", command=self.apply_median_blur_dialog)

        # ---- Canvas Area ----
        self.canvas = tk.Canvas(self.root, bg="lightgray")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

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
        self.brush_size_slider = tk.Scale(control_panel, from_=1, to=20, orient="horizontal", command=self.update_brush_size)
        self.brush_size_slider.set(self.pen_size)
        self.brush_size_slider.pack(pady=(0,10))
        tk.Label(control_panel, text="Brush Color:").pack()
        self.color_btn = tk.Button(control_panel, text="Choose Color", command=self.choose_color)
        self.color_btn.pack(pady=(0,10))
        
        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-z>", lambda e: self.undo_action())
        self.root.bind("<Control-y>", lambda e: self.redo_action())

        # Initialize filter classes
        self.linear_filters = LinearFilters()
        self.nonlinear_filters = NonLinearFilters()
        self.edge_detection = EdgeDetection()
        self.geometric_transforms = GeometricTransforms()

    def pil_to_cv(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def cv_to_pil(self, cv_image):
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    def update_brush_size(self, value):
        self.pen_size = int(float(value))
        self.status.config(text=f"Brush Size: {self.pen_size}")

    def undo_action(self):
        if self.undo_stack:
            item = self.undo_stack.pop()
            self.redo_stack.append(item)
            self.canvas.delete(item['id'])
            self.status.config(text="Undo successful")
        else:
            self.status.config(text="Nothing to undo")

    def redo_action(self):
        if self.redo_stack:
            item = self.redo_stack.pop()
            if item.get('type') == 'emoji':
                obj = self.canvas.create_text(item['x'], item['y'], text=item['emoji'], font=("Arial", item['size']))
                item['id'] = obj
                self.undo_stack.append(item)
            elif item.get('type') == 'line':
                obj = self.canvas.create_line(item['coords'], fill=item['color'], width=item['width'], capstyle="round", smooth=True)
                item['id'] = obj
                self.undo_stack.append(item)
            self.status.config(text="Redo successful")
        else:
            self.status.config(text="Nothing to redo")

    def add_emoji_mode(self):
        emojis = ["😀","😂","😍","😎","🥳","😜","🤩","😭","😡","👍","👀","🎉","💖","🔥","🤖","😇","😏","😱","😴","😈","👻","💩","🙈","🙉","🙊","🐶","🐱","🦄","🍕","🍔","🍟","🍦","🍩","🍉","🍓","🍒","🍇","🍌","🍍","🥑","🥦","🥕","🌈","⭐","⚡","☀️","🌙"]
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
        self.undo_stack.append({'type':'emoji','x':x,'y':y,'emoji':self.selected_emoji,'size':self.emoji_size,'id':obj})
        self.redo_stack.clear()
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files","*.jpg *.png *.jpeg *.bmp *.gif")])
        if path:
            self.file_path = path
            self.image = Image.open(path)
            self.original_image = self.image.copy()
            self.display_image()
            self.status.config(text=f"Opened: {path}")

    def display_image(self):
        if self.image:
            self.canvas.delete("all")
            self.root.update()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.image.size
            padding = 50
            scale = min((canvas_width-padding)/img_width, (canvas_height-padding)/img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            use_auto = messagebox.askyesno("Resize Image", f"Auto-resize to {new_width}x{new_height} to fit canvas?\n\nClick 'No' for custom size.")
            if not use_auto:
                resize_win = tk.Toplevel(self.root)
                resize_win.title("Resize Image")
                resize_win.geometry("300x200")
                tk.Label(resize_win, text="Width:").pack()
                width_scale = tk.Scale(resize_win, from_=int(img_width/4), to=int(img_width*2), orient="horizontal")
                width_scale.set(new_width)
                width_scale.pack()
                tk.Label(resize_win, text="Height:").pack()
                height_scale = tk.Scale(resize_win, from_=int(img_height/4), to=int(img_height*2), orient="horizontal")
                height_scale.set(new_height)
                height_scale.pack()
                def apply_resize():
                    nonlocal new_width, new_height
                    new_width = width_scale.get()
                    new_height = height_scale.get()
                    resize_win.destroy()
                tk.Button(resize_win, text="Apply", command=apply_resize).pack(pady=10)
                resize_win.grab_set()
                self.root.wait_window(resize_win)
            self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.image)
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.tk_image, tags="image")
            self.status.config(text=f"Image loaded: {new_width}x{new_height}")
            
    def reset_to_original(self):
        if hasattr(self, 'original_image') and self.original_image:
            self.image = self.original_image.copy()
            self.display_image()
            self.status.config(text="Reset to original image")
        else:
            messagebox.showinfo("No Original", "No original image to reset to!")

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files","*.png"),("JPEG files","*.jpg")])
        if path:
            try:
                self.root.update()
                x = self.canvas.winfo_rootx()
                y = self.canvas.winfo_rooty()
                w = self.canvas.winfo_width()
                h = self.canvas.winfo_height()
                img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
                img.save(path)
                self.status.config(text=f"Saved: {path}")
                messagebox.showinfo("Success", f"Image saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose drawing color")
        if color[1]:
            self.pen_color = color[1]
            self.color_btn.config(bg=self.pen_color)
            self.status.config(text=f"Selected Color: {self.pen_color}")

    def enable_draw(self):
        self.status.config(text="Drawing mode enabled - Click and drag to draw")

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.pen_size = self.brush_size_slider.get()
            obj = self.canvas.create_line(self.last_x, self.last_y, x, y, fill=self.pen_color, width=self.pen_size, capstyle="round", smooth=True)
            self.undo_stack.append({'type':'line','coords':(self.last_x,self.last_y,x,y),'color':self.pen_color,'width':self.pen_size,'id':obj})
            self.last_x, self.last_y = x, y
            self.redo_stack.clear()

    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        if messagebox.askyesno("Clear Canvas", "Are you sure? This cannot be undone!"):
            self.canvas.delete("all")
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.status.config(text="Canvas cleared")

    def apply_mean_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.linear_filters.mean_filter(cv_img, kernel_size=5)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Mean filter applied (kernel size: 5)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_gaussian_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.linear_filters.gaussian_filter(cv_img, kernel_size=5, sigma=1.5)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Gaussian blur applied (kernel: 5, sigma: 1.5)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_sharpen_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.linear_filters.sharpen_filter(cv_img)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Sharpen filter applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_median_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.nonlinear_filters.median_filter(cv_img, kernel_size=5)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Median filter applied (kernel size: 5)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_sobel(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.edge_detection.sobel_edge(cv_img)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Sobel edge detection applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_prewitt(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.edge_detection.prewitt_edge(cv_img)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Prewitt edge detection applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def apply_laplacian(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            filtered = self.edge_detection.laplacian_edge(cv_img)
            self.image = self.cv_to_pil(filtered)
            self.display_image()
            self.status.config(text="Laplacian edge detection applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def crop_image_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        crop_win = tk.Toplevel(self.root)
        crop_win.title("Crop Image")
        crop_win.geometry("300x250")
        h, w = self.pil_to_cv(self.image).shape[:2]
        tk.Label(crop_win, text=f"Image Size: {w} x {h}").pack(pady=5)
        tk.Label(crop_win, text="X (Left):").pack()
        x_scale = tk.Scale(crop_win, from_=0, to=w-1, orient="horizontal")
        x_scale.pack()
        tk.Label(crop_win, text="Y (Top):").pack()
        y_scale = tk.Scale(crop_win, from_=0, to=h-1, orient="horizontal")
        y_scale.pack()
        tk.Label(crop_win, text="Width:").pack()
        w_scale = tk.Scale(crop_win, from_=1, to=w, orient="horizontal")
        w_scale.set(w//2)
        w_scale.pack()
        tk.Label(crop_win, text="Height:").pack()
        h_scale = tk.Scale(crop_win, from_=1, to=h, orient="horizontal")
        h_scale.set(h//2)
        h_scale.pack()
        def apply_crop():
            try:
                cv_img = self.pil_to_cv(self.image)
                cropped = self.geometric_transforms.crop_image(cv_img, x_scale.get(), y_scale.get(), w_scale.get(), h_scale.get())
                self.image = self.cv_to_pil(cropped)
                self.display_image()
                self.status.config(text="Image cropped")
                crop_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to crop:\n{e}")
        tk.Button(crop_win, text="Apply Crop", command=apply_crop).pack(pady=10)
    
    def resize_image_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        resize_win = tk.Toplevel(self.root)
        resize_win.title("Resize Image")
        resize_win.geometry("300x200")
        w, h = self.image.size
        tk.Label(resize_win, text=f"Current Size: {w} x {h}").pack(pady=5)
        tk.Label(resize_win, text="New Width:").pack()
        width_scale = tk.Scale(resize_win, from_=50, to=w*2, orient="horizontal")
        width_scale.set(w)
        width_scale.pack()
        tk.Label(resize_win, text="New Height:").pack()
        height_scale = tk.Scale(resize_win, from_=50, to=h*2, orient="horizontal")
        height_scale.set(h)
        height_scale.pack()
        def apply_resize():
            try:
                cv_img = self.pil_to_cv(self.image)
                resized = self.geometric_transforms.resize_image(cv_img, width=width_scale.get(), height=height_scale.get())
                self.image = self.cv_to_pil(resized)
                self.display_image()
                self.status.config(text=f"Resized to {width_scale.get()}x{height_scale.get()}")
                resize_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to resize:\n{e}")
        tk.Button(resize_win, text="Apply Resize", command=apply_resize).pack(pady=10)
    
    def rotate_90_degrees(self, k):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            rotated = self.geometric_transforms.rotate_90(cv_img, k)
            self.image = self.cv_to_pil(rotated)
            self.display_image()
            angle = k * 90
            self.status.config(text=f"Rotated {angle}° counter-clockwise")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate:\n{e}")
    
    def rotate_custom_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        rotate_win = tk.Toplevel(self.root)
        rotate_win.title("Custom Rotation")
        rotate_win.geometry("300x200")
        tk.Label(rotate_win, text="Rotation Angle (degrees):").pack(pady=5)
        angle_scale = tk.Scale(rotate_win, from_=-180, to=180, orient="horizontal")
        angle_scale.set(0)
        angle_scale.pack()
        tk.Label(rotate_win, text="Scale:").pack()
        scale_scale = tk.Scale(rotate_win, from_=0.1, to=2.0, resolution=0.1, orient="horizontal")
        scale_scale.set(1.0)
        scale_scale.pack()
        keep_size_var = tk.BooleanVar(value=True)
        tk.Checkbutton(rotate_win, text="Keep original size", variable=keep_size_var).pack()
        def apply_rotation():
            try:
                cv_img = self.pil_to_cv(self.image)
                rotated = self.geometric_transforms.rotate_image(cv_img, angle_scale.get(), scale=scale_scale.get(), keep_size=keep_size_var.get())
                self.image = self.cv_to_pil(rotated)
                self.display_image()
                self.status.config(text=f"Rotated {angle_scale.get()}°")
                rotate_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to rotate:\n{e}")
        tk.Button(rotate_win, text="Apply Rotation", command=apply_rotation).pack(pady=10)
    
    def flip_image(self, flip_code):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            flipped = self.geometric_transforms.flip_image(cv_img, flip_code)
            self.image = self.cv_to_pil(flipped)
            self.display_image()
            flip_type = "Horizontal" if flip_code == 1 else "Vertical"
            self.status.config(text=f"Flipped {flip_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to flip:\n{e}")

    def adjust_brightness_contrast_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        adjust_win = tk.Toplevel(self.root)
        adjust_win.title("Brightness & Contrast")
        adjust_win.geometry("350x200")
        tk.Label(adjust_win, text="Brightness (-127 to 127):").pack(pady=5)
        brightness_scale = tk.Scale(adjust_win, from_=-127, to=127, orient="horizontal")
        brightness_scale.set(0)
        brightness_scale.pack()
        tk.Label(adjust_win, text="Contrast (-127 to 127):").pack(pady=5)
        contrast_scale = tk.Scale(adjust_win, from_=-127, to=127, orient="horizontal")
        contrast_scale.set(0)
        contrast_scale.pack()
        def apply_adjustment():
            try:
                cv_img = self.pil_to_cv(self.image)
                adjusted = adjust_brightness_contrast(cv_img, brightness=brightness_scale.get(), contrast=contrast_scale.get())
                self.image = self.cv_to_pil(adjusted)
                self.display_image()
                self.status.config(text=f"Brightness: {brightness_scale.get()}, Contrast: {contrast_scale.get()}")
                adjust_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to adjust:\n{e}")
        tk.Button(adjust_win, text="Apply", command=apply_adjustment).pack(pady=10)
    
    def adjust_saturation_hue_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        adjust_win = tk.Toplevel(self.root)
        adjust_win.title("Saturation & Hue")
        adjust_win.geometry("350x200")
        tk.Label(adjust_win, text="Saturation (-255 to 255):").pack(pady=5)
        saturation_scale = tk.Scale(adjust_win, from_=-255, to=255, orient="horizontal")
        saturation_scale.set(0)
        saturation_scale.pack()
        tk.Label(adjust_win, text="Hue (-179 to 179):").pack(pady=5)
        hue_scale = tk.Scale(adjust_win, from_=-179, to=179, orient="horizontal")
        hue_scale.set(0)
        hue_scale.pack()
        def apply_adjustment():
            try:
                cv_img = self.pil_to_cv(self.image)
                adjusted = adjust_saturation_hue(cv_img, saturation=saturation_scale.get(), hue=hue_scale.get())
                self.image = self.cv_to_pil(adjusted)
                self.display_image()
                self.status.config(text=f"Saturation: {saturation_scale.get()}, Hue: {hue_scale.get()}")
                adjust_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to adjust:\n{e}")
        tk.Button(adjust_win, text="Apply", command=apply_adjustment).pack(pady=10)
    
    def adjust_gamma_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        gamma_win = tk.Toplevel(self.root)
        gamma_win.title("Gamma Correction")
        gamma_win.geometry("350x150")
        tk.Label(gamma_win, text="Gamma (0.1 to 3.0):").pack(pady=5)
        tk.Label(gamma_win, text="< 1.0 = Brighter | > 1.0 = Darker", font=("Arial", 9)).pack()
        gamma_scale = tk.Scale(gamma_win, from_=0.1, to=3.0, resolution=0.1, orient="horizontal")
        gamma_scale.set(1.0)
        gamma_scale.pack()
        def apply_gamma_correction():
            try:
                cv_img = self.pil_to_cv(self.image)
                adjusted = adjust_gamma(cv_img, gamma=gamma_scale.get())
                self.image = self.cv_to_pil(adjusted)
                self.display_image()
                self.status.config(text=f"Gamma correction applied: {gamma_scale.get()}")
                gamma_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply gamma:\n{e}")
        tk.Button(gamma_win, text="Apply", command=apply_gamma_correction).pack(pady=10)
    
    def apply_histogram_eq(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            equalized = apply_histogram_equalization(cv_img)
            self.image = self.cv_to_pil(equalized)
            self.display_image()
            self.status.config(text="Histogram equalization applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply histogram equalization:\n{e}")
    
    def apply_global_threshold_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        threshold_win = tk.Toplevel(self.root)
        threshold_win.title("Global Threshold")
        threshold_win.geometry("350x150")
        tk.Label(threshold_win, text="Threshold Value (0 to 255):").pack(pady=5)
        threshold_scale = tk.Scale(threshold_win, from_=0, to=255, orient="horizontal")
        threshold_scale.set(127)
        threshold_scale.pack()
        def apply_threshold():
            try:
                cv_img = self.pil_to_cv(self.image)
                thresholded = apply_global_threshold(cv_img, threshold_value=threshold_scale.get())
                thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                self.image = self.cv_to_pil(thresholded_bgr)
                self.display_image()
                self.status.config(text=f"Global threshold applied: {threshold_scale.get()}")
                threshold_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply threshold:\n{e}")
        tk.Button(threshold_win, text="Apply", command=apply_threshold).pack(pady=10)
    
    def apply_adaptive_threshold_fn(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        try:
            cv_img = self.pil_to_cv(self.image)
            thresholded = apply_adaptive_threshold(cv_img)
            thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            self.image = self.cv_to_pil(thresholded_bgr)
            self.display_image()
            self.status.config(text="Adaptive threshold applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply adaptive threshold:\n{e}")
    
    def apply_average_blur_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        blur_win = tk.Toplevel(self.root)
        blur_win.title("Average Blur")
        blur_win.geometry("350x150")
        tk.Label(blur_win, text="Kernel Size (odd numbers only):").pack(pady=5)
        kernel_scale = tk.Scale(blur_win, from_=3, to=31, resolution=2, orient="horizontal")
        kernel_scale.set(5)
        kernel_scale.pack()
        def apply_blur():
            try:
                cv_img = self.pil_to_cv(self.image)
                blurred = apply_average_blur(cv_img, kernel_size=kernel_scale.get())
                self.image = self.cv_to_pil(blurred)
                self.display_image()
                self.status.config(text=f"Average blur applied: kernel {kernel_scale.get()}")
                blur_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply blur:\n{e}")
        tk.Button(blur_win, text="Apply", command=apply_blur).pack(pady=10)
    
    def apply_gaussian_blur_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        blur_win = tk.Toplevel(self.root)
        blur_win.title("Gaussian Blur")
        blur_win.geometry("350x150")
        tk.Label(blur_win, text="Kernel Size (odd numbers only):").pack(pady=5)
        kernel_scale = tk.Scale(blur_win, from_=3, to=31, resolution=2, orient="horizontal")
        kernel_scale.set(5)
        kernel_scale.pack()
        def apply_blur():
            try:
                cv_img = self.pil_to_cv(self.image)
                blurred = apply_gaussian_blur(cv_img, kernel_size=kernel_scale.get())
                self.image = self.cv_to_pil(blurred)
                self.display_image()
                self.status.config(text=f"Gaussian blur applied: kernel {kernel_scale.get()}")
                blur_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply blur:\n{e}")
        tk.Button(blur_win, text="Apply", command=apply_blur).pack(pady=10)
    
    def apply_median_blur_dialog(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        blur_win = tk.Toplevel(self.root)
        blur_win.title("Median Blur")
        blur_win.geometry("350x150")
        tk.Label(blur_win, text="Kernel Size (odd numbers only):").pack(pady=5)
        kernel_scale = tk.Scale(blur_win, from_=3, to=31, resolution=2, orient="horizontal")
        kernel_scale.set(5)
        kernel_scale.pack()
        def apply_blur():
            try:
                cv_img = self.pil_to_cv(self.image)
                blurred = apply_median_blur(cv_img, kernel_size=kernel_scale.get())
                self.image = self.cv_to_pil(blurred)
                self.display_image()
                self.status.config(text=f"Median blur applied: kernel {kernel_scale.get()}")
                blur_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply blur:\n{e}")
        tk.Button(blur_win, text="Apply", command=apply_blur).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartImageEditor(root)
    root.mainloop()