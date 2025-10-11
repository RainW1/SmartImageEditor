import customtkinter as ctk
import tkinter as tk  # ← Make sure this is there for StringVar
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import os
import threading
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import LinearFilters, NonLinearFilters, EdgeDetection, GeometricTransforms
from features.ai_filters import (
    AIColorCorrection,
    BackgroundRemoval,
    StyleTransfer,
    ai_color_correction,
    remove_bg,
)


class Debouncer:
    """Debounce rapid slider changes for performance"""

    def __init__(self, delay=0.15):
        self.delay = delay
        self.timer = None

    def debounce(self, func):
        """Call func after delay, canceling previous calls"""
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.delay, func)
        self.timer.start()

    def cancel(self):
        """Cancel pending call"""
        if self.timer:
            self.timer.cancel()


class ModernImageEditor(ctk.CTk):
    """Modern Image Editor with CustomTkinter"""

    def __init__(self):
        super().__init__()

        # ===== WINDOW SETUP =====
        self.title("🎨 Smart Image Editor - Modern")
        self.geometry("1400x900")

        # Set theme
        ctk.set_appearance_mode("dark")  # "dark" or "light"
        ctk.set_default_color_theme("blue")

        # ===== VARIABLES =====
        self.image = None
        self.original_image = None
        self.true_original_image = None
        self.display_image = None
        self.tk_image = None
        self.file_path = None
        self.preview_debouncer = Debouncer(delay=0.1)
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None
        self.crop_rect_id = None

        # Add resize aspect ratio lock
        self.aspect_ratio_locked = True
        self.original_aspect_ratio = None

        # Temporary adjustment values (for real-time preview)
        self.temp_brightness = 0
        self.temp_contrast = 0
        self.temp_saturation = 0

        # Filter instances
        self.linear_filters = LinearFilters()
        self.nonlinear_filters = NonLinearFilters()
        self.edge_detection = EdgeDetection()
        self.geometric_transforms = GeometricTransforms()

        # ===== HISTORY SYSTEM =====
        self.history = []  # Stack of history states
        self.history_index = -1  # Current position in history
        self.max_history = 20  # Maximum history states
        self.history_enabled = True  # Flag to disable during bulk operations
        
            # ===== DRAWING SYSTEM =====
        self.drawing_mode = False
        self.drawing_tool = "pen"  # pen, line, rectangle, circle, arrow, text, fill
        self.drawing_color = "#FF0000"  # Red default
        self.drawing_size = 3
        self.drawing_fill = False
        self.drawing_start_pos = None
        self.drawing_temp_item = None
        self.text_to_draw = ""
        
        self.drawing_history = []
        self.drawing_history_index = -1

        # ===== AI VARIABLES =====
        self.bg_mode_var = None  # Will be set when AI panel opens

        # ===== KEYBOARD SHORTCUTS =====
        self.bind("<Control-z>", lambda e: self.undo_action())
        self.bind("<Control-y>", lambda e: self.redo_action())
        self.bind("<Control-Shift-z>", lambda e: self.redo_action())  # Alternative redo
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-r>", lambda e: self.reset_to_original())

        # ===== BUILD UI =====
        self.setup_ui()

        # ===== STATUS =====
        self.update_status("Ready - Open an image to start")

    def setup_ui(self):
        """Setup main UI layout"""

        # ===== TOP BAR =====
        self.top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.top_bar.pack(fill="x", padx=0, pady=0)
        self.top_bar.pack_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            self.top_bar, text="🎨 Smart Image Editor", font=("Arial", 20, "bold")
        )
        title_label.pack(side="left", padx=20)

        # Top buttons
        btn_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        btn_frame.pack(side="right", padx=20)

        # Open button
        self.btn_open = ctk.CTkButton(
            btn_frame, text="📁 Open", command=self.open_image, width=100, height=35
        )
        self.btn_open.pack(side="left", padx=5)

        # Save button
        self.btn_save = ctk.CTkButton(
            btn_frame,
            text="💾 Save",
            command=self.save_image,
            width=100,
            height=35,
            fg_color="green",
            hover_color="darkgreen",
        )
        self.btn_save.pack(side="left", padx=5)

        # Theme toggle
        self.theme_switch = ctk.CTkSwitch(
            btn_frame, text="🌙", command=self.toggle_theme, width=50
        )
        self.theme_switch.pack(side="left", padx=10)

        # ===== MAIN CONTAINER =====
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ===== LEFT SIDEBAR =====
        self.sidebar = ctk.CTkFrame(main_container, width=250)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar.pack_propagate(False)

        self.setup_sidebar()

        # ===== CENTER CANVAS =====
        self.canvas_frame = ctk.CTkFrame(main_container)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.setup_canvas()

        # ===== RIGHT CONTROL PANEL =====
        self.control_panel = ctk.CTkScrollableFrame(
            main_container,
            width=320,
            label_text="Control Panel",
            fg_color=("#dbdbdb", "#2b2b2b"),  # ← ADD COLORS
        )
        self.control_panel.pack(side="right", fill="both")  # ← REMOVE pack_propagate

        # Start with empty panel
        self.show_empty_panel()

        # ===== BOTTOM STATUS BAR =====
        self.status_bar = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_bar.pack(fill="x", padx=0, pady=0)
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10)

    def setup_sidebar(self):
        """Setup left sidebar with tool categories"""

        # Title
        title = ctk.CTkLabel(self.sidebar, text="🛠️ Tools", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        # Tool buttons
        tools = [
            ("⚙️ Adjust", self.show_adjust_panel, "#3b82f6"),
            ("🤖 AI Filters", self.show_ai_panel, "#8b5cf6"),
            ("🎭 Filters", self.show_filters_panel, "#06b6d4"),
            ("✏️ Drawing", self.show_drawing_panel, "#ec4899"),
            ("🔄 Transform", self.show_transform_panel, "#f59e0b"),
            ("📊 Analysis", self.show_analysis_panel, "#10b981"),
        ]

        for tool_name, command, color in tools:
            btn = ctk.CTkButton(
                self.sidebar,
                text=tool_name,
                command=command,
                height=45,
                font=("Arial", 14),
                fg_color=color,
                hover_color=self.darken_color(color),
            )
            btn.pack(pady=8, padx=20, fill="x")

        # Separator
        separator = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray")
        separator.pack(pady=20, padx=20, fill="x")

        # Quick actions
        quick_label = ctk.CTkLabel(
            self.sidebar,
            text="Quick Actions",
            font=("Arial", 12, "bold"),
            text_color="gray",
        )
        quick_label.pack(pady=(0, 10))

        quick_actions = [
            ("↺ Undo (Ctrl+Z)", self.undo_action),
            ("↻ Redo (Ctrl+Y)", self.redo_action),
            ("🔄 Reset (Ctrl+R)", self.reset_to_original),
        ]

        for action_name, command in quick_actions:
            btn = ctk.CTkButton(
                self.sidebar,
                text=action_name,
                command=command,
                height=35,
                font=("Arial", 12),
                fg_color="gray30",
                hover_color="gray20",
            )
            btn.pack(pady=4, padx=20, fill="x")

    def setup_canvas(self):
        """Setup center canvas area"""

        # Canvas title
        canvas_title = ctk.CTkLabel(
            self.canvas_frame, text="Canvas", font=("Arial", 14, "bold")
        )
        canvas_title.pack(pady=10)

        # Canvas itself
        self.canvas = ctk.CTkCanvas(
            self.canvas_frame, bg="#1a1a1a", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Bind resize event to update placeholder
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Create placeholder (will be positioned on first configure event)
        self.placeholder_text_id = None
        self.create_placeholder()

    def create_placeholder(self):
        """Create or update placeholder text"""
        # Delete old placeholder if exists
        if self.placeholder_text_id:
            self.canvas.delete(self.placeholder_text_id)

        # Get canvas center
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Create centered text
        self.placeholder_text_id = self.canvas.create_text(
            width // 2,
            height // 2,
            text="📁 Open an image to start editing",
            fill="gray",
            font=("Arial", 16),
            tags="placeholder",
        )


    def on_canvas_resize(self, event):
        """Handle canvas resize event"""
        if self.image is not None:
            # Redraw image to fit new canvas size
            self.display_image_on_canvas()
        else:
            # Update placeholder position
            self.create_placeholder()

    def show_empty_panel(self):
        """Show empty control panel"""
        self.clear_control_panel()

        empty_label = ctk.CTkLabel(
            self.control_panel, text="Select a tool from the sidebar", text_color="gray"
        )
        empty_label.pack(pady=50)

    def clear_control_panel(self):
        """Clear all widgets in control panel"""
        for widget in self.control_panel.winfo_children():
            widget.destroy()

    def show_adjust_panel(self):
        """Show adjustment controls"""
        self.clear_control_panel()

        # Title
        title = ctk.CTkLabel(
            self.control_panel, text="⚙️ Adjustments", font=("Arial", 18, "bold")
        )
        title.pack(pady=20)

        # Info
        info = ctk.CTkLabel(
            self.control_panel, text="Real-time preview enabled", text_color="gray"
        )
        info.pack(pady=(0, 20))

        # ===== BRIGHTNESS SLIDER =====
        brightness_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        brightness_frame.pack(pady=15, padx=20, fill="x")

        # Label and value
        brightness_header = ctk.CTkFrame(brightness_frame, fg_color="transparent")
        brightness_header.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(brightness_header, text="Brightness", font=("Arial", 13)).pack(
            side="left"
        )

        self.brightness_value_label = ctk.CTkLabel(
            brightness_header,
            text="0",
            font=("Arial", 13, "bold"),
            text_color="#3b82f6",
        )
        self.brightness_value_label.pack(side="right")

        # Slider
        self.brightness_slider = ctk.CTkSlider(
            brightness_frame,
            from_=-100,
            to=100,
            number_of_steps=200,
            command=lambda v: self.update_brightness_slider(v),
        )
        self.brightness_slider.set(self.temp_brightness)
        self.brightness_slider.pack(fill="x")

        # ===== CONTRAST SLIDER =====
        contrast_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        contrast_frame.pack(pady=15, padx=20, fill="x")

        contrast_header = ctk.CTkFrame(contrast_frame, fg_color="transparent")
        contrast_header.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(contrast_header, text="Contrast", font=("Arial", 13)).pack(
            side="left"
        )

        self.contrast_value_label = ctk.CTkLabel(
            contrast_header, text="0", font=("Arial", 13, "bold"), text_color="#3b82f6"
        )
        self.contrast_value_label.pack(side="right")

        self.contrast_slider = ctk.CTkSlider(
            contrast_frame,
            from_=-100,
            to=100,
            number_of_steps=200,
            command=lambda v: self.update_contrast_slider(v),
        )
        self.contrast_slider.set(self.temp_contrast)
        self.contrast_slider.pack(fill="x")

        # ===== SATURATION SLIDER =====
        saturation_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        saturation_frame.pack(pady=15, padx=20, fill="x")

        saturation_header = ctk.CTkFrame(saturation_frame, fg_color="transparent")
        saturation_header.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(saturation_header, text="Saturation", font=("Arial", 13)).pack(
            side="left"
        )

        self.saturation_value_label = ctk.CTkLabel(
            saturation_header,
            text="0",
            font=("Arial", 13, "bold"),
            text_color="#3b82f6",
        )
        self.saturation_value_label.pack(side="right")

        self.saturation_slider = ctk.CTkSlider(
            saturation_frame,
            from_=-100,
            to=100,
            number_of_steps=200,
            command=lambda v: self.update_saturation_slider(v),
        )
        self.saturation_slider.set(self.temp_saturation)
        self.saturation_slider.pack(fill="x")

        # ===== ACTION BUTTONS =====
        btn_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        btn_frame.pack(pady=30, padx=20, fill="x")

        apply_btn = ctk.CTkButton(
            btn_frame,
            text="✓ Apply Changes",
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="green",
            hover_color="darkgreen",
            command=self.apply_changes,
        )
        apply_btn.pack(fill="x", pady=5)

        reset_btn = ctk.CTkButton(
            btn_frame,
            text="↺ Reset",
            height=40,
            font=("Arial", 13),
            fg_color="gray30",
            hover_color="gray20",
            command=self.reset_adjustments,
        )
        reset_btn.pack(fill="x", pady=5)

    # Tambahkan helper methods untuk update label
    def update_brightness_slider(self, value):
        """Update brightness value label and preview"""
        int_value = int(value)
        self.brightness_value_label.configure(text=str(int_value))
        self.on_brightness_change(int_value)

    def update_contrast_slider(self, value):
        """Update contrast value label and preview"""
        int_value = int(value)
        self.contrast_value_label.configure(text=str(int_value))
        self.on_contrast_change(int_value)

    def update_saturation_slider(self, value):
        """Update saturation value label and preview"""
        int_value = int(value)
        self.saturation_value_label.configure(text=str(int_value))
        self.on_saturation_change(int_value)

    def create_slider(self, label_text, min_val, max_val, default_val, command):
        """Helper to create labeled slider"""

        frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        frame.pack(pady=15, padx=20, fill="x")

        # Label and value
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 5))

        label = ctk.CTkLabel(header, text=label_text, font=("Arial", 13))
        label.pack(side="left")

        value_label = ctk.CTkLabel(
            header,
            text=str(default_val),
            font=("Arial", 13, "bold"),
            text_color="#3b82f6",
        )
        value_label.pack(side="right")

        # Slider
        slider = ctk.CTkSlider(
            frame,
            from_=min_val,
            to=max_val,
            number_of_steps=abs(max_val - min_val),
            command=lambda v: self.slider_callback(v, value_label, command),
        )
        slider.set(default_val)
        slider.pack(fill="x")

        return slider

    def slider_callback(self, value, value_label, command):
        """Handle slider value change"""
        int_value = int(value)
        value_label.configure(text=str(int_value))
        command(int_value)

    def create_action_buttons(self):
        """Create Apply and Reset buttons"""

        btn_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        btn_frame.pack(pady=30, padx=20, fill="x")

        # Apply button
        apply_btn = ctk.CTkButton(
            btn_frame,
            text="✓ Apply Changes",
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="green",
            hover_color="darkgreen",
            command=self.apply_changes,
        )
        apply_btn.pack(fill="x", pady=5)

        # Reset button
        reset_btn = ctk.CTkButton(
            btn_frame,
            text="↺ Reset",
            height=40,
            font=("Arial", 13),
            fg_color="gray30",
            hover_color="gray20",
            command=self.reset_adjustments,
        )
        reset_btn.pack(fill="x", pady=5)

    # ===== AI PANEL IMPLEMENTATION =====
    # Replace show_ai_panel() method with this complete version

    def show_ai_panel(self):
        """Show AI tools panel with full features"""
        self.clear_control_panel()

        # Title
        title = ctk.CTkLabel(
            self.control_panel, text="🤖 AI Tools", font=("Arial", 18, "bold")
        )
        title.pack(pady=20)

        # ===== AI COLOR CORRECTION SECTION =====
        ai_color_frame = ctk.CTkFrame(self.control_panel)
        ai_color_frame.pack(pady=10, padx=20, fill="x")

        ai_color_title = ctk.CTkLabel(
            ai_color_frame, text="✨ AI Color Correction", font=("Arial", 14, "bold")
        )
        ai_color_title.pack(pady=10)

        # Quick enhance buttons
        quick_enhance_frame = ctk.CTkFrame(ai_color_frame, fg_color="transparent")
        quick_enhance_frame.pack(pady=10, fill="x")

        btn_auto_enhance = ctk.CTkButton(
            quick_enhance_frame,
            text="⚡ Auto Enhance",
            height=35,
            command=self.ai_auto_enhance,
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
        )
        btn_auto_enhance.pack(side="left", padx=5, expand=True, fill="x")

        btn_clahe = ctk.CTkButton(
            quick_enhance_frame,
            text="🎨 CLAHE",
            height=35,
            command=self.ai_clahe_quick,
            fg_color="#6366f1",
            hover_color="#4f46e5",
        )
        btn_clahe.pack(side="left", padx=5, expand=True, fill="x")

        btn_wb = ctk.CTkButton(
            ai_color_frame,
            text="⚖️ White Balance",
            height=35,
            command=self.ai_white_balance,
            fg_color="#3b82f6",
            hover_color="#2563eb",
        )
        btn_wb.pack(pady=5, padx=10, fill="x")

        # Separator
        separator1 = ctk.CTkFrame(ai_color_frame, height=2, fg_color="gray")
        separator1.pack(pady=10, padx=10, fill="x")

        # Advanced controls label
        advanced_label = ctk.CTkLabel(
            ai_color_frame,
            text="Advanced Controls",
            font=("Arial", 12, "bold"),
            text_color="gray",
        )
        advanced_label.pack(pady=(5, 10))

        # Custom color correction button
        btn_custom_color = ctk.CTkButton(
            ai_color_frame,
            text="⚙️ Custom Color Correction...",
            height=40,
            command=self.ai_color_correction_dialog,
            fg_color="#6366f1",
            hover_color="#4f46e5",
        )
        btn_custom_color.pack(pady=5, padx=10, fill="x")

        # ===== BACKGROUND REMOVAL SECTION =====
        bg_frame = ctk.CTkFrame(self.control_panel)
        bg_frame.pack(pady=10, padx=20, fill="x")

        bg_title = ctk.CTkLabel(
            bg_frame, text="🗑️ Background Removal", font=("Arial", 14, "bold")
        )
        bg_title.pack(pady=10)

        # Check if rembg is available
        if BackgroundRemoval.is_available():
            bg_desc = ctk.CTkLabel(
                bg_frame,
                text="AI-powered background removal",
                text_color="gray",
                font=("Arial", 10),
            )
            bg_desc.pack(pady=5)

            # Mode selection
            mode_label = ctk.CTkLabel(
                bg_frame, text="Select Mode:", font=("Arial", 11, "bold")
            )
            mode_label.pack(pady=(10, 5))

            self.bg_mode_var = ctk.StringVar(value="General Mode")

            modes = [
                ("👤 Portrait", "Portrait Mode"),
                ("🎯 General", "General Mode"),
                ("📦 Product", "Product Mode"),
                ("🎨 Anime", "Anime Mode"),
            ]

            for mode_text, mode_value in modes:
                radio = ctk.CTkRadioButton(
                    bg_frame,
                    text=mode_text,
                    variable=self.bg_mode_var,
                    value=mode_value,
                    font=("Arial", 11),
                )
                radio.pack(pady=2, padx=20, anchor="w")

            # Remove background button
            btn_remove_bg = ctk.CTkButton(
                bg_frame,
                text="🚀 Remove Background",
                height=45,
                command=self.ai_remove_background,
                fg_color="#ef4444",
                hover_color="#dc2626",
                font=("Arial", 12, "bold"),
            )
            btn_remove_bg.pack(pady=15, padx=10, fill="x")

        else:
            # Show installation instructions
            warning_label = ctk.CTkLabel(
                bg_frame,
                text="⚠️ Not Available",
                font=("Arial", 12, "bold"),
                text_color="#ef4444",
            )
            warning_label.pack(pady=10)

            info_label = ctk.CTkLabel(
                bg_frame,
                text="Install rembg to enable:\npip install rembg",
                text_color="gray",
                font=("Arial", 10),
                justify="center",
            )
            info_label.pack(pady=5)

        # ===== STYLE TRANSFER SECTION (Optional) =====
        if StyleTransfer.is_available():
            style_frame = ctk.CTkFrame(self.control_panel)
            style_frame.pack(pady=10, padx=20, fill="x")

            style_title = ctk.CTkLabel(
                style_frame, text="🎨 Neural Style Transfer", font=("Arial", 14, "bold")
            )
            style_title.pack(pady=10)

            style_desc = ctk.CTkLabel(
                style_frame,
                text="Apply artistic styles (slow!)",
                text_color="gray",
                font=("Arial", 10),
            )
            style_desc.pack(pady=5)

            btn_style = ctk.CTkButton(
                style_frame,
                text="🖼️ Style Transfer...",
                height=40,
                command=self.ai_style_transfer_dialog,
                fg_color="#9333ea",
                hover_color="#7e22ce",
            )
            btn_style.pack(pady=10, padx=10, fill="x")

    # ===== AI COLOR CORRECTION METHODS =====

    def ai_auto_enhance(self):
        """Quick auto enhance with default settings"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            self.update_status("⏳ Applying AI auto enhancement...")
            self.update()

            # Apply full color correction with optimal settings
            enhanced = ai_color_correction(
                self.image.copy(),
                clip_limit=3.0,
                tile_size=8,
                gamma=1.0,
                brightness=0,
                contrast=0,
                wb_toggle=True,
            )

            self.image = enhanced
            self.original_image = enhanced.copy()
            self.add_to_history("AI Auto Enhance")
            self.display_image_on_canvas()

        except Exception as e:
            messagebox.showerror("Error", f"AI enhancement failed:\n{e}")
            self.update_status("❌ Enhancement failed")

    def ai_clahe_quick(self):
        """Quick CLAHE enhancement"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            self.update_status("⏳ Applying CLAHE...")
            self.update()

            enhanced = AIColorCorrection.apply_clahe(
                self.image.copy(), clip_limit=3.0, tile_size=8
            )

            self.image = enhanced
            self.original_image = enhanced.copy()
            self.add_to_history("CLAHE Enhancement")
            self.display_image_on_canvas()

        except Exception as e:
            messagebox.showerror("Error", f"CLAHE failed:\n{e}")
            self.update_status("❌ CLAHE failed")

    def ai_white_balance(self):
        """Quick white balance correction"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            self.update_status("⏳ Applying white balance...")
            self.update()

            balanced = AIColorCorrection.white_balance(self.image.copy())

            self.image = balanced
            self.original_image = balanced.copy()
            self.add_to_history("White Balance")
            self.display_image_on_canvas()

        except Exception as e:
            messagebox.showerror("Error", f"White balance failed:\n{e}")
            self.update_status("❌ White balance failed")

    def ai_color_correction_dialog(self):
        """Show custom AI color correction dialog with all settings"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("AI Color Correction - Custom Settings")
        dialog.geometry("450x600")
        dialog.grab_set()  # Make modal

        # Title
        title = ctk.CTkLabel(
            dialog, text="⚙️ Custom AI Color Correction", font=("Arial", 18, "bold")
        )
        title.pack(pady=20)

        # Settings frame
        settings_frame = ctk.CTkScrollableFrame(dialog, width=400, height=400)
        settings_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # CLAHE Clip Limit
        ctk.CTkLabel(
            settings_frame, text="CLAHE Clip Limit (1.0 - 5.0):", font=("Arial", 12)
        ).pack(pady=(10, 5), anchor="w")
        clip_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        clip_frame.pack(fill="x", pady=5)

        clip_value_label = ctk.CTkLabel(
            clip_frame, text="3.0", font=("Arial", 12, "bold"), text_color="#3b82f6"
        )
        clip_value_label.pack(side="right")

        clip_slider = ctk.CTkSlider(
            settings_frame, from_=1.0, to=5.0, number_of_steps=40
        )
        clip_slider.set(3.0)
        clip_slider.pack(fill="x", pady=5)
        clip_slider.configure(
            command=lambda v: clip_value_label.configure(text=f"{v:.1f}")
        )

        # Tile Size
        ctk.CTkLabel(
            settings_frame, text="Tile Grid Size (2 - 16):", font=("Arial", 12)
        ).pack(pady=(10, 5), anchor="w")
        tile_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        tile_frame.pack(fill="x", pady=5)

        tile_value_label = ctk.CTkLabel(
            tile_frame, text="8", font=("Arial", 12, "bold"), text_color="#3b82f6"
        )
        tile_value_label.pack(side="right")

        tile_slider = ctk.CTkSlider(settings_frame, from_=2, to=16, number_of_steps=14)
        tile_slider.set(8)
        tile_slider.pack(fill="x", pady=5)
        tile_slider.configure(
            command=lambda v: tile_value_label.configure(text=str(int(v)))
        )

        # Gamma
        ctk.CTkLabel(
            settings_frame, text="Gamma (0.5 - 3.0):", font=("Arial", 12)
        ).pack(pady=(10, 5), anchor="w")
        ctk.CTkLabel(
            settings_frame,
            text="< 1.0 = Brighter | > 1.0 = Darker",
            font=("Arial", 9),
            text_color="gray",
        ).pack(anchor="w")
        gamma_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        gamma_frame.pack(fill="x", pady=5)

        gamma_value_label = ctk.CTkLabel(
            gamma_frame, text="1.0", font=("Arial", 12, "bold"), text_color="#3b82f6"
        )
        gamma_value_label.pack(side="right")

        gamma_slider = ctk.CTkSlider(
            settings_frame, from_=0.5, to=3.0, number_of_steps=50
        )
        gamma_slider.set(1.0)
        gamma_slider.pack(fill="x", pady=5)
        gamma_slider.configure(
            command=lambda v: gamma_value_label.configure(text=f"{v:.1f}")
        )

        # Brightness
        ctk.CTkLabel(
            settings_frame, text="Brightness (-50 to 50):", font=("Arial", 12)
        ).pack(pady=(10, 5), anchor="w")
        brightness_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        brightness_frame.pack(fill="x", pady=5)

        brightness_value_label = ctk.CTkLabel(
            brightness_frame, text="0", font=("Arial", 12, "bold"), text_color="#3b82f6"
        )
        brightness_value_label.pack(side="right")

        brightness_slider = ctk.CTkSlider(
            settings_frame, from_=-50, to=50, number_of_steps=100
        )
        brightness_slider.set(0)
        brightness_slider.pack(fill="x", pady=5)
        brightness_slider.configure(
            command=lambda v: brightness_value_label.configure(text=str(int(v)))
        )

        # Contrast
        ctk.CTkLabel(
            settings_frame, text="Contrast (-50 to 50):", font=("Arial", 12)
        ).pack(pady=(10, 5), anchor="w")
        contrast_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        contrast_frame.pack(fill="x", pady=5)

        contrast_value_label = ctk.CTkLabel(
            contrast_frame, text="0", font=("Arial", 12, "bold"), text_color="#3b82f6"
        )
        contrast_value_label.pack(side="right")

        contrast_slider = ctk.CTkSlider(
            settings_frame, from_=-50, to=50, number_of_steps=100
        )
        contrast_slider.set(0)
        contrast_slider.pack(fill="x", pady=5)
        contrast_slider.configure(
            command=lambda v: contrast_value_label.configure(text=str(int(v)))
        )

        # White Balance toggle
        wb_var = ctk.BooleanVar(value=True)
        wb_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Apply Auto White Balance",
            variable=wb_var,
            font=("Arial", 12),
        )
        wb_checkbox.pack(pady=15)

        # Apply button
        def apply_custom_correction():
            try:
                dialog.destroy()
                self.update_status("⏳ Applying custom AI color correction...")
                self.update()

                corrected = ai_color_correction(
                    self.image.copy(),
                    clip_limit=clip_slider.get(),
                    tile_size=int(tile_slider.get()),
                    gamma=gamma_slider.get(),
                    brightness=int(brightness_slider.get()),
                    contrast=int(contrast_slider.get()),
                    wb_toggle=wb_var.get(),
                )

                self.image = corrected
                self.original_image = corrected.copy()
                self.add_to_history("AI Custom Color Correction")
                self.display_image_on_canvas()

            except Exception as e:
                messagebox.showerror("Error", f"Color correction failed:\n{e}")
                self.update_status("❌ Correction failed")

        btn_apply = ctk.CTkButton(
            dialog,
            text="✨ Apply Color Correction",
            height=45,
            command=apply_custom_correction,
            fg_color="green",
            hover_color="darkgreen",
            font=("Arial", 13, "bold"),
        )
        btn_apply.pack(pady=20, padx=20, fill="x")

    # ===== BACKGROUND REMOVAL METHOD =====

    def ai_remove_background(self):
        """Remove background using AI with selected mode (threaded)"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        if not BackgroundRemoval.is_available():
            messagebox.showerror(
                "Not Available",
                "Background removal requires 'rembg' package.\n\nInstall with:\npip install rembg",
            )
            return

        # Get selected mode
        mode = self.bg_mode_var.get()

        # Confirm (it takes time)
        if not messagebox.askyesno(
            "Remove Background",
            f"Remove background using {mode}?\n\n⏳ This may take 10-30 seconds...\n\nContinue?",
        ):
            return

        try:
            # Save current image for processing
            image_to_process = self.image.copy()

            # Show progress in status
            self.update_status(f"⏳ Removing background ({mode})... Please wait...")

            # Create progress window
            progress_window = ctk.CTkToplevel(self)
            progress_window.title("Processing...")
            progress_window.geometry("400x200")
            progress_window.grab_set()

            # Make it stay on top
            progress_window.attributes("-topmost", True)

            progress_label = ctk.CTkLabel(
                progress_window,
                text=f"🗑️ Removing background...\nMode: {mode}",
                font=("Arial", 14, "bold"),
            )
            progress_label.pack(pady=20)

            progress_bar = ctk.CTkProgressBar(progress_window, width=350)
            progress_bar.pack(pady=10)
            progress_bar.set(0)

            status_label = ctk.CTkLabel(
                progress_window,
                text="Processing with AI... This may take 10-30 seconds",
                font=("Arial", 11),
                text_color="gray",
            )
            status_label.pack(pady=10)

            cancel_label = ctk.CTkLabel(
                progress_window,
                text="⚠️ Do not close this window",
                font=("Arial", 10),
                text_color="#ef4444",
            )
            cancel_label.pack(pady=5)

            progress_window.update()

            # Variables for thread communication
            result_container = {"result": None, "error": None, "done": False}

            # Worker function that runs in background thread
            def remove_bg_worker():
                try:
                    result = remove_bg(image_to_process, mode=mode)
                    result_container["result"] = result
                    result_container["done"] = True
                except Exception as e:
                    result_container["error"] = str(e)
                    result_container["done"] = True

            # Start worker thread
            import threading

            worker_thread = threading.Thread(target=remove_bg_worker, daemon=True)
            worker_thread.start()

            # Animate progress bar while waiting
            progress_value = 0
            while not result_container["done"]:
                # Update progress bar (indeterminate style)
                progress_value += 0.02
                if progress_value > 1.0:
                    progress_value = 0
                progress_bar.set(progress_value)

                # Update window
                progress_window.update()

                # Small delay
                time.sleep(0.05)

            # Set progress to complete
            progress_bar.set(1.0)
            progress_window.update()
            time.sleep(0.3)  # Show completion briefly

            # Close progress window
            progress_window.destroy()

            # Check for errors
            if result_container["error"]:
                raise Exception(result_container["error"])

            # Update image with result
            self.image = result_container["result"]
            self.original_image = self.image.copy()
            self.add_to_history(f"Remove BG ({mode})")
            self.display_image_on_canvas()

            self.update_status(
                f"✅ Background removed ({mode}) | {self.get_history_position()}"
            )
            messagebox.showinfo(
                "Success", f"Background removed successfully!\n\nMode: {mode}"
            )

        except Exception as e:
            if "progress_window" in locals():
                try:
                    progress_window.destroy()
                except:
                    pass
            messagebox.showerror("Error", f"Background removal failed:\n{e}")
            self.update_status("❌ Background removal failed")

    # ===== STYLE TRANSFER METHOD =====

    def ai_style_transfer_dialog(self):
        """Show style transfer dialog with full implementation"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        
        if not StyleTransfer.is_available():
            messagebox.showerror(
                "Not Available",
                "Style transfer requires PyTorch.\n\nInstall with:\npip install torch torchvision"
            )
            return
        
        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Neural Style Transfer")
        dialog.geometry("500x500")
        dialog.grab_set()
        
        # Title
        title = ctk.CTkLabel(
            dialog,
            text="🎨 Neural Style Transfer",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=20)
        
        # Warning
        warning = ctk.CTkLabel(
            dialog,
            text="⚠️ This process takes 30-60 seconds and uses VGG19 neural network",
            font=("Arial", 11),
            text_color="#ef4444",
            wraplength=400
        )
        warning.pack(pady=10)
        
        # Style image selection
        style_frame = ctk.CTkFrame(dialog)
        style_frame.pack(pady=20, padx=20, fill="x")
        
        style_label = ctk.CTkLabel(
            style_frame,
            text="1️⃣ Select Style Image:",
            font=("Arial", 13, "bold")
        )
        style_label.pack(pady=10)
        
        style_path_var = tk.StringVar(value="No style image selected")
        style_path_label = ctk.CTkLabel(
            style_frame,
            textvariable=style_path_var,
            font=("Arial", 10),
            text_color="gray",
            wraplength=400
        )
        style_path_label.pack(pady=5)
        
        style_image_var = [None]  # Store style image
        
        def select_style_image():
            path = filedialog.askopenfilename(
                title="Select Style Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if path:
                try:
                    style_img = cv2.imread(path)
                    if style_img is not None:
                        style_image_var[0] = style_img
                        style_path_var.set(f"✅ Selected: {os.path.basename(path)}")
                    else:
                        messagebox.showerror("Error", "Failed to load style image!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load:\n{e}")
        
        select_btn = ctk.CTkButton(
            style_frame,
            text="📁 Browse Style Image...",
            height=40,
            command=select_style_image,
            fg_color="#6366f1",
            hover_color="#4f46e5"
        )
        select_btn.pack(pady=10, padx=20, fill="x")
        
        # Intensity slider
        intensity_frame = ctk.CTkFrame(dialog)
        intensity_frame.pack(pady=20, padx=20, fill="x")
        
        intensity_label = ctk.CTkLabel(
            intensity_frame,
            text="2️⃣ Style Intensity (0.0 - 1.0):",
            font=("Arial", 13, "bold")
        )
        intensity_label.pack(pady=10)
        
        intensity_desc = ctk.CTkLabel(
            intensity_frame,
            text="Lower = Keep original | Higher = More stylized",
            font=("Arial", 10),
            text_color="gray"
        )
        intensity_desc.pack(pady=5)
        
        intensity_value_frame = ctk.CTkFrame(intensity_frame, fg_color="transparent")
        intensity_value_frame.pack(fill="x", pady=5)
        
        intensity_value_label = ctk.CTkLabel(
            intensity_value_frame,
            text="0.5",
            font=("Arial", 12, "bold"),
            text_color="#3b82f6"
        )
        intensity_value_label.pack(side="right", padx=10)
        
        intensity_slider = ctk.CTkSlider(
            intensity_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20
        )
        intensity_slider.set(0.5)
        intensity_slider.pack(fill="x", padx=10, pady=5)
        intensity_slider.configure(command=lambda v: intensity_value_label.configure(text=f"{v:.2f}"))
        
        # Apply button
        def apply_style_transfer():
            if style_image_var[0] is None:
                messagebox.showwarning("No Style", "Please select a style image first!")
                return
            
            if not messagebox.askyesno(
                "Confirm",
                "⏳ Style transfer will take 30-60 seconds.\n\n"
                "The window may freeze - this is normal.\n\n"
                "Continue?"
            ):
                return
            
            try:
                dialog.destroy()
                
                # Show progress
                self.update_status("⏳ Applying neural style transfer... This takes 30-60 seconds...")
                self.update()
                
                # Create progress window
                progress_window = ctk.CTkToplevel(self)
                progress_window.title("Processing...")
                progress_window.geometry("450x200")
                progress_window.grab_set()
                progress_window.attributes('-topmost', True)
                
                progress_label = ctk.CTkLabel(
                    progress_window,
                    text="🎨 Neural Style Transfer\nProcessing with VGG19...",
                    font=("Arial", 14, "bold")
                )
                progress_label.pack(pady=20)
                
                progress_bar = ctk.CTkProgressBar(progress_window, width=400)
                progress_bar.pack(pady=10)
                progress_bar.set(0)
                
                status_label = ctk.CTkLabel(
                    progress_window,
                    text="This may take 30-60 seconds...\nWindow may freeze - please wait!",
                    font=("Arial", 11),
                    text_color="gray"
                )
                status_label.pack(pady=10)
                
                warning_label = ctk.CTkLabel(
                    progress_window,
                    text="⚠️ Do NOT close this window!",
                    font=("Arial", 10, "bold"),
                    text_color="#ef4444"
                )
                warning_label.pack(pady=5)
                
                progress_window.update()
                
                # Variables for thread
                result_container = {'result': None, 'error': None, 'done': False}
                
                def style_transfer_worker():
                    try:
                        result = StyleTransfer.apply_style_transfer(
                            self.image.copy(),
                            style_image_var[0],
                            intensity=intensity_slider.get(),
                            num_steps=100
                        )
                        result_container['result'] = result
                        result_container['done'] = True
                    except Exception as e:
                        result_container['error'] = str(e)
                        result_container['done'] = True
                
                # Start worker thread
                import threading
                worker_thread = threading.Thread(target=style_transfer_worker, daemon=True)
                worker_thread.start()
                
                # Animate progress
                progress_value = 0
                while not result_container['done']:
                    progress_value += 0.01
                    if progress_value > 1.0:
                        progress_value = 0
                    progress_bar.set(progress_value)
                    progress_window.update()
                    time.sleep(0.05)
                
                # Complete
                progress_bar.set(1.0)
                progress_window.update()
                time.sleep(0.3)
                progress_window.destroy()
                
                # Check for errors
                if result_container['error']:
                    raise Exception(result_container['error'])
                
                # Update image
                self.image = result_container['result']
                self.original_image = self.image.copy()
                self.add_to_history("Neural Style Transfer")
                self.display_image_on_canvas()
                
                messagebox.showinfo("Success", "Style transfer applied successfully!")
                
            except Exception as e:
                if 'progress_window' in locals():
                    try:
                        progress_window.destroy()
                    except:
                        pass
                messagebox.showerror("Error", f"Style transfer failed:\n{e}")
                self.update_status("❌ Style transfer failed")
        
        apply_btn = ctk.CTkButton(
            dialog,
            text="🚀 Apply Style Transfer",
            height=50,
            command=apply_style_transfer,
            fg_color="#9333ea",
            hover_color="#7e22ce",
            font=("Arial", 13, "bold")
        )
        apply_btn.pack(pady=20, padx=20, fill="x")
        
    def show_drawing_panel(self):
        """Show drawing and annotation panel"""
        self.clear_control_panel()
        
        # Disable crop mode if active
        if self.crop_mode:
            self.exit_crop_mode()
        
        # Title
        title = ctk.CTkLabel(
            self.control_panel,
            text="✏️ Drawing & Annotation",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=20)
        
        if self.image is None:
            info_label = ctk.CTkLabel(
                self.control_panel,
                text="Open an image to start drawing",
                text_color="gray",
                font=("Arial", 12)
            )
            info_label.pack(pady=50)
            return
        
        # ===== DRAWING TOOLS =====
        tools_frame = ctk.CTkFrame(self.control_panel)
        tools_frame.pack(pady=10, padx=20, fill="x")
        
        tools_title = ctk.CTkLabel(
            tools_frame,
            text="🎨 Drawing Tools",
            font=("Arial", 14, "bold")
        )
        tools_title.pack(pady=10)
        
        # Tool selection buttons
        tools = [
            ("✏️ Pen", "pen"),
            ("📏 Line", "line"),
            ("⬜ Rect", "rectangle"),  # ← Shortened text
            ("⭕ Circle", "circle"),
            ("➡️ Arrow", "arrow"),
            ("📝 Text", "text"),
            ("🪣 Fill", "fill"),
        ]
            
        # Create 2 rows: 4 tools in row 1, 3 in row 2
        row1_frame = ctk.CTkFrame(tools_frame, fg_color="transparent")
        row1_frame.pack(fill="x", pady=5, padx=5)

        row2_frame = ctk.CTkFrame(tools_frame, fg_color="transparent")
        row2_frame.pack(fill="x", pady=5, padx=5)

        self.tool_buttons = {}

        for i, (tool_name, tool_id) in enumerate(tools):
            # First 4 go to row 1, rest to row 2
            parent = row1_frame if i < 4 else row2_frame
            
            btn = ctk.CTkButton(
                parent,
                text=tool_name,
                height=40,  # ← Slightly taller
                width=85,   # ← Slightly narrower to fit 4
                command=lambda t=tool_id: self.select_drawing_tool(t),
                fg_color="#ec4899" if tool_id == self.drawing_tool else "gray30",
                hover_color="#db2777",
                font=("Arial", 11)  # ← Smaller font
            )
            btn.pack(side="left", padx=2, pady=2)
            self.tool_buttons[tool_id] = btn
        
        # ===== COLOR PICKER =====
        color_frame = ctk.CTkFrame(self.control_panel)
        color_frame.pack(pady=10, padx=20, fill="x")
        
        color_title = ctk.CTkLabel(
            color_frame,
            text="🎨 Color",
            font=("Arial", 14, "bold")
        )
        color_title.pack(pady=10)
        
        # Color display and picker
        color_display_frame = ctk.CTkFrame(color_frame, fg_color="transparent")
        color_display_frame.pack(fill="x", pady=5)
        
        self.color_display = ctk.CTkButton(
            color_display_frame,
            text="",
            width=60,
            height=40,
            fg_color=self.drawing_color,
            hover_color=self.drawing_color,
            border_width=2,
            border_color="white"
        )
        self.color_display.pack(side="left", padx=10)
        
        color_picker_btn = ctk.CTkButton(
            color_display_frame,
            text="🎨 Choose Color",
            height=40,
            command=self.choose_drawing_color
        )
        color_picker_btn.pack(side="left", padx=10, fill="x", expand=True)
        
        # Quick colors
        quick_colors_frame = ctk.CTkFrame(color_frame, fg_color="transparent")
        quick_colors_frame.pack(fill="x", pady=10)
        
        quick_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
            "#FF00FF", "#00FFFF", "#FFFFFF", "#000000"
        ]
        
        for color in quick_colors:
            btn = ctk.CTkButton(
                quick_colors_frame,
                text="",
                width=30,
                height=30,
                fg_color=color,
                hover_color=color,
                border_width=1,
                border_color="gray",
                command=lambda c=color: self.set_drawing_color(c)
            )
            btn.pack(side="left", padx=2)
        
        # ===== SIZE/THICKNESS =====
        size_frame = ctk.CTkFrame(self.control_panel)
        size_frame.pack(pady=10, padx=20, fill="x")
        
        size_title = ctk.CTkLabel(
            size_frame,
            text="📏 Size/Thickness",
            font=("Arial", 14, "bold")
        )
        size_title.pack(pady=10)
        
        size_value_frame = ctk.CTkFrame(size_frame, fg_color="transparent")
        size_value_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(size_value_frame, text="Size:", font=("Arial", 11)).pack(side="left", padx=10)
        
        self.size_value_label = ctk.CTkLabel(
            size_value_frame,
            text=str(self.drawing_size),
            font=("Arial", 11, "bold"),
            text_color="#3b82f6"
        )
        self.size_value_label.pack(side="right", padx=10)
        
        self.size_slider = ctk.CTkSlider(
            size_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            command=self.update_drawing_size
        )
        self.size_slider.set(self.drawing_size)
        self.size_slider.pack(fill="x", padx=10, pady=5)
        
        # ===== OPTIONS =====
        options_frame = ctk.CTkFrame(self.control_panel)
        options_frame.pack(pady=10, padx=20, fill="x")
        
        options_title = ctk.CTkLabel(
            options_frame,
            text="⚙️ Options",
            font=("Arial", 14, "bold")
        )
        options_title.pack(pady=10)
        
        # Fill checkbox (for shapes)
        self.fill_var = ctk.BooleanVar(value=False)
        fill_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Fill Shapes",
            variable=self.fill_var,
            font=("Arial", 12)
        )
        fill_checkbox.pack(pady=5)
        
        # Text input (for text tool)
        text_label = ctk.CTkLabel(
            options_frame,
            text="Text to Add:",
            font=("Arial", 11)
        )
        text_label.pack(pady=(10, 5))
        
        self.text_entry = ctk.CTkEntry(
            options_frame,
            placeholder_text="Enter text here...",
            height=35
        )
        self.text_entry.pack(pady=5, padx=10, fill="x")
        
        # ===== DRAWING CONTROLS =====
        controls_frame = ctk.CTkFrame(self.control_panel)
        controls_frame.pack(pady=10, padx=20, fill="x")
        
        controls_title = ctk.CTkLabel(
            controls_frame,
            text="🎮 Controls",
            font=("Arial", 14, "bold")
        )
        controls_title.pack(pady=10)
        
        # Enable/Disable drawing
        self.drawing_toggle_btn = ctk.CTkButton(
            controls_frame,
            text="🖌️ Enable Drawing",
            height=45,
            command=self.toggle_drawing_mode,
            fg_color="green",
            hover_color="darkgreen",
            font=("Arial", 13, "bold")
        )
        self.drawing_toggle_btn.pack(pady=10, padx=10, fill="x")
        
        # Clear all drawings
        clear_btn = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear All Drawings",
            height=40,
            command=self.clear_all_drawings,
            fg_color="#ef4444",
            hover_color="#dc2626"
        )
        clear_btn.pack(pady=5, padx=10, fill="x")
        
        # Apply drawings permanently
        apply_btn = ctk.CTkButton(
            controls_frame,
            text="✅ Apply to Image",
            height=40,
            command=self.apply_drawings_to_image,
            fg_color="#10b981",
            hover_color="#059669"
        )
        apply_btn.pack(pady=5, padx=10, fill="x")


    # ===== DRAWING TOOL METHODS =====

    def select_drawing_tool(self, tool):
        """Select drawing tool"""
        self.drawing_tool = tool
        
        # Update button colors
        for tool_id, btn in self.tool_buttons.items():
            if tool_id == tool:
                btn.configure(fg_color="#ec4899")
            else:
                btn.configure(fg_color="gray30")
        
        self.update_status(f"✏️ Selected: {tool.capitalize()} tool")

    def choose_drawing_color(self):
        """Choose drawing color"""
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Choose Drawing Color", initialcolor=self.drawing_color)
        if color[1]:
            self.set_drawing_color(color[1])

    def set_drawing_color(self, color):
        """Set drawing color"""
        self.drawing_color = color
        self.color_display.configure(fg_color=color, hover_color=color)
        self.update_status(f"🎨 Color: {color}")

    def update_drawing_size(self, value):
        """Update drawing size"""
        self.drawing_size = int(value)
        self.size_value_label.configure(text=str(self.drawing_size))

    def toggle_drawing_mode(self):
        """Toggle drawing mode on/off"""
        self.drawing_mode = not self.drawing_mode
        
        if self.drawing_mode:
            # Enable drawing
            self.drawing_toggle_btn.configure(
                text="🛑 Disable Drawing",
                fg_color="#ef4444",
                hover_color="#dc2626"
            )
            
            # Replace shortcuts with drawing-specific undo/redo
            self.unbind("<Control-z>")
            self.unbind("<Control-y>")
            self.unbind("<Control-Shift-z>")
            
            self.bind("<Control-z>", lambda e: self.undo_drawing())
            self.bind("<Control-y>", lambda e: self.redo_drawing())
            self.bind("<Control-Shift-z>", lambda e: self.redo_drawing())
            
            # Initialize drawing history
            self.drawing_history = []
            self.drawing_history_index = -1
            self.add_drawing_to_history()  # Save initial empty state
            
            # Bind mouse events
            self.canvas.bind("<ButtonPress-1>", self.on_drawing_start)
            self.canvas.bind("<B1-Motion>", self.on_drawing_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_drawing_end)
            
            self.update_status(f"✏️ Drawing mode ON - Ctrl+Z/Y for drawing undo/redo")
        else:
            # Disable drawing
            self.drawing_toggle_btn.configure(
                text="🖌️ Enable Drawing",
                fg_color="green",
                hover_color="darkgreen"
            )
            
            # Restore normal undo/redo
            self.unbind("<Control-z>")
            self.unbind("<Control-y>")
            self.unbind("<Control-Shift-z>")
            
            self.bind("<Control-z>", lambda e: self.undo_action())
            self.bind("<Control-y>", lambda e: self.redo_action())
            self.bind("<Control-Shift-z>", lambda e: self.redo_action())
            
            # Unbind mouse events
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            
            self.update_status("✏️ Drawing mode OFF")
            
    def on_drawing_start(self, event):
        """Start drawing"""
        self.drawing_start_pos = (event.x, event.y)
        
        if self.drawing_tool == "pen":
            # Start freehand drawing
            pass
        elif self.drawing_tool == "text":
            # Place text
            self.place_text(event.x, event.y)
        elif self.drawing_tool == "fill":
            # Fill area
            self.fill_area(event.x, event.y)

    def on_drawing_drag(self, event):
        """Handle dragging while drawing"""
        if not self.drawing_start_pos:
            return
        
        x1, y1 = self.drawing_start_pos
        x2, y2 = event.x, event.y
        
        if self.drawing_tool == "pen":
            # Freehand drawing
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=self.drawing_color,
                width=self.drawing_size,
                capstyle="round",
                smooth=True,
                tags="drawing"
            )
            self.drawing_start_pos = (x2, y2)
            
        elif self.drawing_tool in ["line", "rectangle", "circle", "arrow"]:
            # Remove previous temp item
            if self.drawing_temp_item:
                self.canvas.delete(self.drawing_temp_item)
            
            # Draw temp shape
            if self.drawing_tool == "line":
                self.drawing_temp_item = self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=self.drawing_color,
                    width=self.drawing_size,
                    tags="drawing"
                )
            elif self.drawing_tool == "rectangle":
                if self.fill_var.get():
                    self.drawing_temp_item = self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.drawing_color,
                        outline=self.drawing_color,
                        width=self.drawing_size,
                        tags="drawing"
                    )
                else:
                    self.drawing_temp_item = self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline=self.drawing_color,
                        width=self.drawing_size,
                        tags="drawing"
                    )
            elif self.drawing_tool == "circle":
                if self.fill_var.get():
                    self.drawing_temp_item = self.canvas.create_oval(
                        x1, y1, x2, y2,
                        fill=self.drawing_color,
                        outline=self.drawing_color,
                        width=self.drawing_size,
                        tags="drawing"
                    )
                else:
                    self.drawing_temp_item = self.canvas.create_oval(
                        x1, y1, x2, y2,
                        outline=self.drawing_color,
                        width=self.drawing_size,
                        tags="drawing"
                    )
            elif self.drawing_tool == "arrow":
                self.drawing_temp_item = self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=self.drawing_color,
                    width=self.drawing_size,
                    arrow="last",
                    arrowshape=(16, 20, 6),
                    tags="drawing"
                )

    def on_drawing_end(self, event):
        """End drawing"""
        self.drawing_temp_item = None
        self.drawing_start_pos = None
        
        self.add_drawing_to_history()

    def place_text(self, x, y):
        """Place text on canvas"""
        text = self.text_entry.get()
        if not text:
            messagebox.showwarning("No Text", "Please enter text in the text box first!")
            return
        
        # Calculate font size based on drawing size
        font_size = 10 + (self.drawing_size * 2)
        
        self.canvas.create_text(
            x, y,
            text=text,
            fill=self.drawing_color,
            font=("Arial", font_size, "bold"),
            tags="drawing"
        )
        
        self.update_status(f"📝 Text added: '{text}'")

    def fill_area(self, x, y):
        """Fill area with color (flood fill simulation)"""
        # Create filled rectangle at click position
        size = 20 + (self.drawing_size * 5)
        self.canvas.create_oval(
            x - size, y - size, x + size, y + size,
            fill=self.drawing_color,
            outline=self.drawing_color,
            tags="drawing"
        )
        self.update_status(f"🪣 Fill applied at ({x}, {y})")

    def clear_all_drawings(self):
        """Clear all drawings from canvas"""
        if messagebox.askyesno("Clear Drawings", "Remove all drawings from canvas?"):
            self.canvas.delete("drawing")
            self.update_status("🗑️ All drawings cleared")

    def apply_drawings_to_image(self):
        """Apply drawings permanently to image"""
        if self.image is None:
            return
        
        if not messagebox.askyesno(
            "Apply Drawings",
            "Apply all drawings permanently to the image?\n\nThis cannot be undone after saving."
        ):
            return
        
        try:
            # Capture canvas with drawings
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            
            import pyscreenshot as ImageGrab
            canvas_img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            
            # Convert to OpenCV format
            canvas_array = np.array(canvas_img)
            canvas_cv = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2BGR)
            
            # Resize to match image size
            img_h, img_w = self.image.shape[:2]
            canvas_resized = cv2.resize(canvas_cv, (img_w, img_h))
            
            # Update image
            self.image = canvas_resized
            self.original_image = canvas_resized.copy()
            
            # Clear drawings and redisplay
            self.canvas.delete("drawing")
            self.add_to_history("Applied Drawings")
            self.display_image_on_canvas()
            
            messagebox.showinfo("Success", "Drawings applied to image!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply drawings:\n{e}")    
            
    def add_drawing_to_history(self):
        """Save current canvas drawing state"""
        # Get all items with 'drawing' tag
        drawing_items = self.canvas.find_withtag("drawing")
        
        # Save item data
        items_data = []
        for item in drawing_items:
            item_type = self.canvas.type(item)
            coords = self.canvas.coords(item)
            config = {}
            
            # Save relevant config
            if item_type == "line":
                config = {
                    'fill': self.canvas.itemcget(item, 'fill'),
                    'width': self.canvas.itemcget(item, 'width'),
                }
            elif item_type in ["rectangle", "oval"]:
                config = {
                    'outline': self.canvas.itemcget(item, 'outline'),
                    'fill': self.canvas.itemcget(item, 'fill'),
                    'width': self.canvas.itemcget(item, 'width'),
                }
            elif item_type == "text":
                config = {
                    'fill': self.canvas.itemcget(item, 'fill'),
                    'font': self.canvas.itemcget(item, 'font'),
                    'text': self.canvas.itemcget(item, 'text'),
                }
            
            items_data.append({
                'type': item_type,
                'coords': coords,
                'config': config
            })
        
        # Remove future history
        if self.drawing_history_index < len(self.drawing_history) - 1:
            self.drawing_history = self.drawing_history[:self.drawing_history_index + 1]
        
        # Add to history
        self.drawing_history.append(items_data)
        self.drawing_history_index += 1

    def undo_drawing(self):
        """Undo last drawing action"""
        if self.drawing_history_index > 0:
            self.drawing_history_index -= 1
            self.restore_drawing_state()
            self.update_status(f"↶ Drawing Undo | {self.drawing_history_index + 1}/{len(self.drawing_history)}")
        else:
            self.update_status("Nothing to undo in drawings")

    def redo_drawing(self):
        """Redo drawing action"""
        if self.drawing_history_index < len(self.drawing_history) - 1:
            self.drawing_history_index += 1
            self.restore_drawing_state()
            self.update_status(f"↷ Drawing Redo | {self.drawing_history_index + 1}/{len(self.drawing_history)}")
        else:
            self.update_status("Nothing to redo in drawings")

    def restore_drawing_state(self):
        """Restore canvas to saved drawing state"""
        # Clear current drawings
        self.canvas.delete("drawing")
        
        # Restore from history
        if 0 <= self.drawing_history_index < len(self.drawing_history):
            items_data = self.drawing_history[self.drawing_history_index]
            
            for item_data in items_data:
                item_type = item_data['type']
                coords = item_data['coords']
                config = item_data['config']
                
                if item_type == "line":
                    self.canvas.create_line(*coords, tags="drawing", **config)
                elif item_type == "rectangle":
                    self.canvas.create_rectangle(*coords, tags="drawing", **config)
                elif item_type == "oval":
                    self.canvas.create_oval(*coords, tags="drawing", **config)
                elif item_type == "text":
                    self.canvas.create_text(*coords, tags="drawing", **config)                
            
    # ===== ROTATE METHODS =====

    def rotate_image(self, angle):
        """Rotate image by specified angle (90, 180, 270)"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            # Convert angle to k for np.rot90
            if angle == 90:
                k = 1  # CCW
            elif angle == 180:
                k = 2
            elif angle == 270:
                k = 3  # CW
            else:
                return

            self.image = np.rot90(self.image, k)
            self.original_image = self.image.copy()

            # Add to history
            self.add_to_history(f"Rotate {angle}°")

            self.display_image_on_canvas()

            # Refresh panel to update size display
            self.show_transform_panel()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate:\n{e}")

    def apply_custom_rotation(self):
        """Apply custom rotation angle"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            angle_str = self.rotate_angle_entry.get()
            if not angle_str:
                messagebox.showwarning("No Angle", "Please enter rotation angle!")
                return

            angle = float(angle_str)

            # Get image center
            h, w = self.image.shape[:2]
            center = (w // 2, h // 2)

            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new size to fit entire rotated image
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Apply rotation
            self.image = cv2.warpAffine(self.image, M, (new_w, new_h))
            self.original_image = self.image.copy()
            self.add_to_history(f"Rotate {angle}°")  # ← ADD THIS
            self.display_image_on_canvas()
            self.update_status(f"✅ Rotated {angle}°")

            # Refresh panel
            self.show_transform_panel()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate:\n{e}")

    # ===== FLIP METHODS =====

    def flip_image(self, flip_code):
        """Flip image (0=vertical, 1=horizontal)"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            self.image = cv2.flip(self.image, flip_code)
            self.original_image = self.image.copy()

            flip_type = "horizontal" if flip_code == 1 else "vertical"

            self.add_to_history(f"Flip {flip_type.capitalize()}")

            self.display_image_on_canvas()
            self.update_status(f"✅ Flipped {flip_type}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to flip:\n{e}")

    # ===== CROP METHODS =====

    def toggle_crop_mode(self):
        """Toggle crop mode on/off"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        self.crop_mode = not self.crop_mode

        if self.crop_mode:
            # Enable crop mode
            self.crop_button.configure(
                text="✓ Apply Crop", fg_color="green", hover_color="darkgreen"
            )
            self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
            self.canvas.bind("<B1-Motion>", self.on_crop_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
            self.update_status("✂️ Crop mode: Click and drag to select area")
        else:
            # Apply crop if rectangle exists
            if self.crop_rect_id:
                self.apply_crop()
            else:
                # Just exit crop mode
                self.exit_crop_mode()

    def on_crop_start(self, event):
        """Start crop selection"""
        self.crop_start = (event.x, event.y)

        # Clear previous rectangle
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)

        # Create new rectangle
        self.crop_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="cyan", width=2, dash=(5, 5)
        )

    def on_crop_drag(self, event):
        """Update crop rectangle while dragging"""
        if self.crop_start and self.crop_rect_id:
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y

            # Update rectangle
            self.canvas.coords(self.crop_rect_id, x1, y1, x2, y2)

    def on_crop_end(self, event):
        """Finish crop selection"""
        if self.crop_start:
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y

            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            self.crop_rect = (x1, y1, x2, y2)
            self.update_status(
                f"✂️ Area selected: {x2-x1}×{y2-y1} px - Click Apply to crop"
            )

    def apply_crop(self):
        """Apply the crop to the image"""
        if not self.crop_rect:
            messagebox.showwarning("No Selection", "Please select an area to crop!")
            return

        try:
            x1, y1, x2, y2 = self.crop_rect

            # Get actual image coordinates (canvas might be scaled)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            h, w = self.image.shape[:2]

            # Calculate scale
            scale = min((canvas_width - 40) / w, (canvas_height - 40) / h)

            # Calculate offset (image is centered)
            img_x = (canvas_width - (w * scale)) / 2
            img_y = (canvas_height - (h * scale)) / 2

            # Convert canvas coords to image coords
            img_x1 = int((x1 - img_x) / scale)
            img_y1 = int((y1 - img_y) / scale)
            img_x2 = int((x2 - img_x) / scale)
            img_y2 = int((y2 - img_y) / scale)

            # Clamp to image bounds
            img_x1 = max(0, min(img_x1, w))
            img_y1 = max(0, min(img_y1, h))
            img_x2 = max(0, min(img_x2, w))
            img_y2 = max(0, min(img_y2, h))

            # Check if valid crop
            if img_x2 <= img_x1 or img_y2 <= img_y1:
                messagebox.showerror("Invalid Crop", "Crop area is too small!")
                return

            # Crop image
            self.image = self.image[img_y1:img_y2, img_x1:img_x2]
            self.original_image = self.image.copy()

            # Add to history
            self.add_to_history(f"Crop to {img_x2-img_x1}×{img_y2-img_y1}px")

            # Exit crop mode and refresh
            self.exit_crop_mode()
            self.display_image_on_canvas()
            self.update_status(f"✅ Cropped to {img_x2-img_x1}×{img_y2-img_y1} px")

            # Refresh panel to update size
            self.show_transform_panel()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to crop:\n{e}")

    def exit_crop_mode(self):
        """Exit crop mode"""
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None

        # Remove rectangle
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

        # Unbind events
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        # Reset button
        self.crop_button.configure(
            text="Start Crop Mode", fg_color="#06b6d4", hover_color="#0891b2"
        )
        self.update_status("✂️ Crop mode exited")

    # ===== RESIZE METHODS =====

    def on_width_change(self, event):
        """Handle width entry change"""
        if not self.aspect_lock_var.get():
            return

        try:
            width = int(self.resize_width_entry.get())
            if self.original_aspect_ratio:
                height = int(width / self.original_aspect_ratio)
                self.resize_height_entry.delete(0, "end")
                self.resize_height_entry.insert(0, str(height))
        except ValueError:
            pass

    def on_height_change(self, event):
        """Handle height entry change"""
        if not self.aspect_lock_var.get():
            return

        try:
            height = int(self.resize_height_entry.get())
            if self.original_aspect_ratio:
                width = int(height * self.original_aspect_ratio)
                self.resize_width_entry.delete(0, "end")
                self.resize_width_entry.insert(0, str(width))
        except ValueError:
            pass

    def toggle_aspect_lock(self):
        """Toggle aspect ratio lock"""
        locked = self.aspect_lock_var.get()
        status = "locked" if locked else "unlocked"
        self.update_status(f"🔒 Aspect ratio {status}")

    def apply_resize(self):
        """Apply resize to image"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        try:
            width_str = self.resize_width_entry.get()
            height_str = self.resize_height_entry.get()

            if not width_str or not height_str:
                messagebox.showwarning(
                    "Invalid Input", "Please enter both width and height!"
                )
                return

            new_width = int(width_str)
            new_height = int(height_str)

            if new_width <= 0 or new_height <= 0:
                messagebox.showerror(
                    "Invalid Size", "Width and height must be positive!"
                )
                return

            # Resize image
            self.image = cv2.resize(
                self.image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )
            self.original_image = self.image.copy()

            # Add to history
            self.add_to_history(f"Resize to {new_width}×{new_height}px")

            # Update aspect ratio
            self.original_aspect_ratio = new_width / new_height

            self.display_image_on_canvas()
            self.update_status(f"✅ Resized to {new_width}×{new_height} px")

            # Refresh panel
            self.show_transform_panel()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resize:\n{e}")

    def show_filters_panel(self):
        """Show filters panel"""
        self.clear_control_panel()

        title = ctk.CTkLabel(
            self.control_panel, text="🎭 Filters", font=("Arial", 18, "bold")
        )
        title.pack(pady=20)

        # Filter categories
        filters = [
            (
                "Blur",
                [
                    ("Mean Blur", self.apply_mean_filter),
                    ("Gaussian Blur", self.apply_gaussian_filter),
                    ("Median Blur", self.apply_median_filter),
                ],
            ),
            (
                "Edge Detection",
                [
                    ("Sobel", self.apply_sobel),
                    ("Prewitt", self.apply_prewitt),
                    ("Laplacian", self.apply_laplacian),
                ],
            ),
            (
                "Enhancement",
                [
                    ("Sharpen", self.apply_sharpen_filter),
                ],
            ),
        ]

        for category, filter_list in filters:
            # Category label
            cat_label = ctk.CTkLabel(
                self.control_panel,
                text=category,
                font=("Arial", 13, "bold"),
                text_color="gray",
            )
            cat_label.pack(pady=(15, 5), anchor="w", padx=20)

            # Filter buttons
            for filter_name, command in filter_list:
                btn = ctk.CTkButton(
                    self.control_panel, text=filter_name, height=35, command=command
                )
                btn.pack(pady=3, padx=20, fill="x")

    def show_transform_panel(self):
        """Show transform panel with all transformation tools"""
        self.clear_control_panel()

        # Title
        title = ctk.CTkLabel(
            self.control_panel, text="🔄 Transform", font=("Arial", 18, "bold")
        )
        title.pack(pady=20)

        # ===== ROTATE SECTION =====
        rotate_frame = ctk.CTkFrame(self.control_panel)
        rotate_frame.pack(pady=10, padx=20, fill="x")

        rotate_label = ctk.CTkLabel(
            rotate_frame, text="🔄 Rotate", font=("Arial", 14, "bold")
        )
        rotate_label.pack(pady=10)

        # Quick rotate buttons
        quick_rotate_frame = ctk.CTkFrame(rotate_frame, fg_color="transparent")
        quick_rotate_frame.pack(pady=5, fill="x")

        btn_90cw = ctk.CTkButton(
            quick_rotate_frame,
            text="↻ 90° CW",
            height=35,
            command=lambda: self.rotate_image(270),  # 270 = clockwise 90
            width=100,
        )
        btn_90cw.pack(side="left", padx=5, expand=True)

        btn_90ccw = ctk.CTkButton(
            quick_rotate_frame,
            text="↺ 90° CCW",
            height=35,
            command=lambda: self.rotate_image(90),
            width=100,
        )
        btn_90ccw.pack(side="left", padx=5, expand=True)

        btn_180 = ctk.CTkButton(
            rotate_frame,
            text="↻ 180°",
            height=35,
            command=lambda: self.rotate_image(180),
        )
        btn_180.pack(pady=5, padx=10, fill="x")

        # Custom rotate
        custom_rotate_frame = ctk.CTkFrame(rotate_frame, fg_color="transparent")
        custom_rotate_frame.pack(pady=10, fill="x")

        ctk.CTkLabel(
            custom_rotate_frame, text="Custom Angle:", font=("Arial", 12)
        ).pack(side="left", padx=5)

        self.rotate_angle_entry = ctk.CTkEntry(
            custom_rotate_frame, width=60, placeholder_text="0"
        )
        self.rotate_angle_entry.pack(side="left", padx=5)

        ctk.CTkLabel(custom_rotate_frame, text="°", font=("Arial", 12)).pack(
            side="left"
        )

        btn_custom_rotate = ctk.CTkButton(
            rotate_frame,
            text="Apply Custom Rotation",
            height=35,
            command=self.apply_custom_rotation,
            fg_color="#f59e0b",
            hover_color="#d97706",
        )
        btn_custom_rotate.pack(pady=5, padx=10, fill="x")

        # ===== FLIP SECTION =====
        flip_frame = ctk.CTkFrame(self.control_panel)
        flip_frame.pack(pady=10, padx=20, fill="x")

        flip_label = ctk.CTkLabel(
            flip_frame, text="🔃 Flip", font=("Arial", 14, "bold")
        )
        flip_label.pack(pady=10)

        flip_buttons_frame = ctk.CTkFrame(flip_frame, fg_color="transparent")
        flip_buttons_frame.pack(pady=5, fill="x")

        btn_flip_h = ctk.CTkButton(
            flip_buttons_frame,
            text="↔️ Horizontal",
            height=35,
            command=lambda: self.flip_image(1),
            width=120,
        )
        btn_flip_h.pack(side="left", padx=5, expand=True)

        btn_flip_v = ctk.CTkButton(
            flip_buttons_frame,
            text="↕️ Vertical",
            height=35,
            command=lambda: self.flip_image(0),
            width=120,
        )
        btn_flip_v.pack(side="left", padx=5, expand=True)

        # ===== CROP SECTION =====
        crop_frame = ctk.CTkFrame(self.control_panel)
        crop_frame.pack(pady=10, padx=20, fill="x")

        crop_label = ctk.CTkLabel(crop_frame, text="✂️ Crop", font=("Arial", 14, "bold"))
        crop_label.pack(pady=10)

        crop_info = ctk.CTkLabel(
            crop_frame,
            text="Click and drag on image to select area",
            text_color="gray",
            font=("Arial", 10),
        )
        crop_info.pack(pady=5)

        self.crop_button = ctk.CTkButton(
            crop_frame,
            text="Start Crop Mode",
            height=40,
            command=self.toggle_crop_mode,
            fg_color="#06b6d4",
            hover_color="#0891b2",
        )
        self.crop_button.pack(pady=5, padx=10, fill="x")

        # ===== RESIZE SECTION =====
        resize_frame = ctk.CTkFrame(self.control_panel)
        resize_frame.pack(pady=10, padx=20, fill="x")

        resize_label = ctk.CTkLabel(
            resize_frame, text="📏 Resize", font=("Arial", 14, "bold")
        )
        resize_label.pack(pady=10)

        # Current size display
        if self.image is not None:
            h, w = self.image.shape[:2]
            current_size = ctk.CTkLabel(
                resize_frame,
                text=f"Current: {w} × {h} px",
                text_color="gray",
                font=("Arial", 11),
            )
            current_size.pack(pady=5)

            # Store aspect ratio
            self.original_aspect_ratio = w / h
        else:
            self.original_aspect_ratio = 1.0

        # Width input
        width_frame = ctk.CTkFrame(resize_frame, fg_color="transparent")
        width_frame.pack(pady=5, fill="x")

        ctk.CTkLabel(width_frame, text="Width:", font=("Arial", 12)).pack(
            side="left", padx=5
        )

        self.resize_width_entry = ctk.CTkEntry(
            width_frame, width=80, placeholder_text="Width"
        )
        self.resize_width_entry.pack(side="left", padx=5)
        self.resize_width_entry.bind("<KeyRelease>", self.on_width_change)

        if self.image is not None:
            h, w = self.image.shape[:2]
            self.resize_width_entry.insert(0, str(w))

        ctk.CTkLabel(width_frame, text="px", font=("Arial", 11)).pack(side="left")

        # Height input
        height_frame = ctk.CTkFrame(resize_frame, fg_color="transparent")
        height_frame.pack(pady=5, fill="x")

        ctk.CTkLabel(height_frame, text="Height:", font=("Arial", 12)).pack(
            side="left", padx=5
        )

        self.resize_height_entry = ctk.CTkEntry(
            height_frame, width=80, placeholder_text="Height"
        )
        self.resize_height_entry.pack(side="left", padx=5)
        self.resize_height_entry.bind("<KeyRelease>", self.on_height_change)

        if self.image is not None:
            h, w = self.image.shape[:2]
            self.resize_height_entry.insert(0, str(h))

        ctk.CTkLabel(height_frame, text="px", font=("Arial", 11)).pack(side="left")

        # Aspect ratio lock
        self.aspect_lock_var = ctk.BooleanVar(value=True)
        self.aspect_lock_checkbox = ctk.CTkCheckBox(
            resize_frame,
            text="🔒 Lock Aspect Ratio",
            variable=self.aspect_lock_var,
            command=self.toggle_aspect_lock,
        )
        self.aspect_lock_checkbox.pack(pady=10)

        # Apply resize button
        btn_apply_resize = ctk.CTkButton(
            resize_frame,
            text="Apply Resize",
            height=40,
            command=self.apply_resize,
            fg_color="green",
            hover_color="darkgreen",
        )
        btn_apply_resize.pack(pady=5, padx=10, fill="x")

    def show_analysis_panel(self):
        """Show image analysis panel"""
        self.clear_control_panel()
        
        # Title
        title = ctk.CTkLabel(
            self.control_panel,
            text="📊 Image Analysis",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=20)
        
        if self.image is None:
            # No image loaded
            info_label = ctk.CTkLabel(
                self.control_panel,
                text="Open an image to see analysis",
                text_color="gray",
                font=("Arial", 12)
            )
            info_label.pack(pady=50)
            return
        
        # ===== IMAGE INFORMATION =====
        info_frame = ctk.CTkFrame(self.control_panel)
        info_frame.pack(pady=10, padx=20, fill="x")
        
        info_title = ctk.CTkLabel(
            info_frame,
            text="📋 Image Information",
            font=("Arial", 14, "bold")
        )
        info_title.pack(pady=10)
        
        # Get image info
        h, w = self.image.shape[:2]
        channels = self.image.shape[2] if len(self.image.shape) == 3 else 1
        
        # Calculate file size (approximate)
        file_size_mb = (self.image.nbytes / (1024 * 1024))
        
        # Create info text
        info_data = [
            ("Resolution", f"{w} × {h} pixels"),
            ("Aspect Ratio", f"{w/h:.2f}:1"),
            ("Channels", f"{channels} ({'BGR' if channels == 3 else 'BGRA' if channels == 4 else 'Grayscale'})"),
            ("Color Depth", "8-bit per channel"),
            ("Memory Size", f"{file_size_mb:.2f} MB"),
        ]
        
        for label, value in info_data:
            info_row = ctk.CTkFrame(info_frame, fg_color="transparent")
            info_row.pack(fill="x", padx=10, pady=3)
            
            ctk.CTkLabel(
                info_row,
                text=f"{label}:",
                font=("Arial", 11),
                anchor="w"
            ).pack(side="left")
            
            ctk.CTkLabel(
                info_row,
                text=value,
                font=("Arial", 11, "bold"),
                text_color="#3b82f6",
                anchor="e"
            ).pack(side="right")
        
        # ===== COLOR STATISTICS =====
        stats_frame = ctk.CTkFrame(self.control_panel)
        stats_frame.pack(pady=10, padx=20, fill="x")
        
        stats_title = ctk.CTkLabel(
            stats_frame,
            text="🎨 Color Statistics",
            font=("Arial", 14, "bold")
        )
        stats_title.pack(pady=10)
        
        # Calculate statistics
        if len(self.image.shape) == 3 and self.image.shape[2] >= 3:
            b_mean = int(np.mean(self.image[:,:,0]))
            g_mean = int(np.mean(self.image[:,:,1]))
            r_mean = int(np.mean(self.image[:,:,2]))
            
            brightness = int((r_mean + g_mean + b_mean) / 3)
            
            stats_data = [
                ("Average Blue", f"{b_mean}"),
                ("Average Green", f"{g_mean}"),
                ("Average Red", f"{r_mean}"),
                ("Overall Brightness", f"{brightness}"),
            ]
        else:
            mean_val = int(np.mean(self.image))
            stats_data = [
                ("Average Intensity", f"{mean_val}"),
            ]
        
        for label, value in stats_data:
            stat_row = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_row.pack(fill="x", padx=10, pady=3)
            
            ctk.CTkLabel(
                stat_row,
                text=f"{label}:",
                font=("Arial", 11),
                anchor="w"
            ).pack(side="left")
            
            ctk.CTkLabel(
                stat_row,
                text=value,
                font=("Arial", 11, "bold"),
                text_color="#10b981",
                anchor="e"
            ).pack(side="right")
        
        # ===== HISTOGRAM BUTTON =====
        histogram_btn = ctk.CTkButton(
            stats_frame,
            text="📈 Show Histogram",
            height=40,
            command=self.show_histogram_window,
            fg_color="#10b981",
            hover_color="#059669"
        )
        histogram_btn.pack(pady=15, padx=10, fill="x")
        
        # ===== FREQUENCY ANALYSIS BUTTON =====
        freq_frame = ctk.CTkFrame(self.control_panel)
        freq_frame.pack(pady=10, padx=20, fill="x")
        
        freq_title = ctk.CTkLabel(
            freq_frame,
            text="🌊 Frequency Domain",
            font=("Arial", 14, "bold")
        )
        freq_title.pack(pady=10)
        
        freq_btn = ctk.CTkButton(
            freq_frame,
            text="📊 Show Frequency Analysis",
            height=40,
            command=self.show_frequency_window,
            fg_color="#6366f1",
            hover_color="#4f46e5"
        )
        freq_btn.pack(pady=10, padx=10, fill="x")

    def show_histogram_window(self):
        """Show histogram in new window"""
        if self.image is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create window
            hist_window = ctk.CTkToplevel(self)
            hist_window.title("Image Histogram")
            hist_window.geometry("800x600")
            
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.patch.set_facecolor('#2b2b2b')
            
            if len(self.image.shape) == 3 and self.image.shape[2] >= 3:
                colors = ('b', 'g', 'r')
                labels = ('Blue', 'Green', 'Red')
                
                # Individual channel histograms
                for i, (color, label) in enumerate(zip(colors, labels)):
                    ax = axes[0, i] if i < 2 else axes[1, 0]
                    hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color)
                    ax.set_title(f'{label} Channel', color='white')
                    ax.set_xlim([0, 256])
                    ax.set_facecolor('#1a1a1a')
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
                # Combined histogram
                ax = axes[1, 1]
                for i, (color, label) in enumerate(zip(colors, labels)):
                    hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color, label=label, alpha=0.7)
                ax.set_title('Combined Channels', color='white')
                ax.set_xlim([0, 256])
                ax.legend()
                ax.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                # Grayscale histogram
                ax = axes[0, 0]
                hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
                ax.plot(hist, color='gray')
                ax.set_title('Intensity', color='white')
                ax.set_xlim([0, 256])
                ax.set_facecolor('#1a1a1a')
                
                # Hide other subplots
                for i in range(1, 4):
                    fig.delaxes(axes.flatten()[i])
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show histogram:\n{e}")


    def show_frequency_window(self):
        """Show frequency domain analysis"""
        if self.image is None:
            return
        
        try:
            from features.frequency_domain import FrequencyDomainAnalysis
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create window
            freq_window = ctk.CTkToplevel(self)
            freq_window.title("Frequency Domain Analysis")
            freq_window.geometry("1200x800")
            
            # Show analysis
            self.update_status("⏳ Computing frequency analysis...")
            self.update()
            
            fig = FrequencyDomainAnalysis.visualize_frequency_analysis(
                self.image,
                title="Frequency Domain Analysis"
            )
            
            # Dark theme for matplotlib
            fig.patch.set_facecolor('#2b2b2b')
            for ax in fig.get_axes():
                ax.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('white')
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, freq_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            self.update_status("✅ Frequency analysis displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show frequency analysis:\n{e}")

    # ===== IMAGE HANDLING =====

    def open_image(self):
        """Open image file"""
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if path:
            self.file_path = path
            img = cv2.imread(path)
            if img is not None:
                self.true_original_image = img.copy()
                self.original_image = img.copy()
                self.image = img.copy()

                # Clear history when opening new image
                self.clear_history()

                # Add initial state to history
                self.add_to_history("Open Image")

                self.display_image_on_canvas()
                self.update_status(
                    f"Opened: {os.path.basename(path)} | {self.get_history_position()}"
                )
            else:
                messagebox.showerror("Error", "Failed to load image!")

    def display_image_on_canvas(self):
        """Display image on canvas"""
        if self.image is None:
            return

        # Remove placeholder
        self.canvas.delete("placeholder")
        self.placeholder_text_id = None  # ← ADD THIS

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 600

        h, w = img_rgb.shape[:2]
        scale = min((canvas_width - 40) / w, (canvas_height - 40) / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to PIL and Tk
        img_pil = Image.fromarray(img_resized)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        # Display on canvas
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.tk_image, anchor="center")

    def save_image(self):
        """Save current image"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, self.image)
            self.update_status(f"Saved: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Image saved to:\n{path}")

    def on_brightness_change(self, value):
        """Handle brightness slider change"""
        self.temp_brightness = int(value)
        self.apply_preview()

    def on_contrast_change(self, value):
        """Handle contrast slider change"""
        self.temp_contrast = int(value)
        self.apply_preview()

    def on_saturation_change(self, value):
        """Handle saturation slider change"""
        self.temp_saturation = int(value)
        self.apply_preview()

    def apply_preview(self):
        """Apply real-time preview with debouncing"""
        if self.original_image is None:
            return

        # Debounce to avoid too many updates
        self.preview_debouncer.debounce(self._apply_preview_internal)

    def _apply_preview_internal(self):
        """Internal method that actually applies the preview"""
        if self.original_image is None:
            return

        try:
            # Start with original image
            preview = self.original_image.copy()

            # Apply brightness adjustment
            if self.temp_brightness != 0:
                preview = cv2.convertScaleAbs(
                    preview, alpha=1.0, beta=self.temp_brightness
                )

            # Apply contrast adjustment
            if self.temp_contrast != 0:
                alpha = 1.0 + (self.temp_contrast / 100.0)
                preview = cv2.convertScaleAbs(preview, alpha=alpha, beta=0)

            # Apply saturation adjustment
            if self.temp_saturation != 0:
                # Convert to HSV
                hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV).astype(np.float32)

                # Adjust saturation channel
                saturation_scale = 1.0 + (self.temp_saturation / 100.0)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_scale

                # Clip to valid range
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

                # Convert back to BGR
                preview = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Update displayed image
            self.image = preview
            self.display_image_on_canvas()

            # Update status
            status_msg = f"Preview: "
            if self.temp_brightness != 0:
                status_msg += f"B={self.temp_brightness:+d} "
            if self.temp_contrast != 0:
                status_msg += f"C={self.temp_contrast:+d} "
            if self.temp_saturation != 0:
                status_msg += f"S={self.temp_saturation:+d} "

            if status_msg == "Preview: ":
                status_msg = "Preview: No changes"

            self.update_status(status_msg)

        except Exception as e:
            print(f"⚠️ Preview error: {e}")
            self.update_status(f"Preview error: {e}")

    def apply_changes(self):
        """Apply all adjustments permanently"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return

        if (
            self.temp_brightness == 0
            and self.temp_contrast == 0
            and self.temp_saturation == 0
        ):
            messagebox.showinfo("No Changes", "No adjustments to apply!")
            return

        # Current preview becomes the new original
        self.original_image = self.image.copy()

        # Build description
        desc_parts = []
        if self.temp_brightness != 0:
            desc_parts.append(f"B{self.temp_brightness:+d}")
        if self.temp_contrast != 0:
            desc_parts.append(f"C{self.temp_contrast:+d}")
        if self.temp_saturation != 0:
            desc_parts.append(f"S{self.temp_saturation:+d}")

        description = "Adjust: " + ", ".join(desc_parts)

        # Add to history
        self.add_to_history(description)

        # Reset temp values
        self.temp_brightness = 0
        self.temp_contrast = 0
        self.temp_saturation = 0

        # Refresh the adjust panel to reset sliders
        self.show_adjust_panel()

        messagebox.showinfo("Success", "Adjustments applied successfully!")

    def reset_adjustments(self):
        """Reset all adjustments to zero"""
        self.temp_brightness = 0
        self.temp_contrast = 0
        self.temp_saturation = 0

        # Reset to original image
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image_on_canvas()

        # Refresh panel to reset sliders
        self.show_adjust_panel()

        self.update_status("↺ Adjustments reset")

    def reset_to_original(self):
        """Reset to TRUE original image (first opened)"""
        if self.true_original_image is not None:
            # Confirm if changes exist
            if not np.array_equal(self.image, self.true_original_image):
                if not messagebox.askyesno(
                    "Reset to Original",
                    "This will discard all changes and reset to the original image.\n\nContinue?",
                ):
                    return

            self.image = self.true_original_image.copy()
            self.original_image = self.true_original_image.copy()
            self.display_image_on_canvas()
            self.update_status("↺ Reset to original image")

            # Reset adjustment values
            self.temp_brightness = 0
            self.temp_contrast = 0
            self.temp_saturation = 0

            # If in adjust panel, refresh it
            # (This will reset sliders to 0)

        else:
            messagebox.showinfo("No Image", "No original image to reset to!")

    # ===== FILTERS =====

    def apply_mean_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.linear_filters.mean_filter(self.image, 5)
        self.add_to_history("Mean Blur Filter")
        self.display_image_on_canvas()
        self.update_status("Mean filter applied")

    def apply_gaussian_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.linear_filters.gaussian_filter(self.image, 5, 1.5)
        self.add_to_history("Gaussian Blur Filter")
        self.display_image_on_canvas()
        self.update_status("Gaussian filter applied")

    def apply_median_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.nonlinear_filters.median_filter(self.image, 5)
        self.add_to_history("Median Blur Filter")
        self.display_image_on_canvas()
        self.update_status("Median filter applied")

    def apply_sharpen_filter(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.linear_filters.sharpen_filter(self.image)
        self.add_to_history("Sharpen Filter")
        self.display_image_on_canvas()
        self.update_status("Sharpen filter applied")

    def apply_sobel(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.edge_detection.sobel_edge(self.image)
        self.add_to_history("Sobel Edge Detection")
        self.display_image_on_canvas()
        self.update_status("Sobel edge detection applied")

    def apply_prewitt(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.edge_detection.prewitt_edge(self.image)
        self.add_to_history("Prewitt Edge Detection")
        self.display_image_on_canvas()
        self.update_status("Prewitt edge detection applied")

    def apply_laplacian(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        self.image = self.edge_detection.laplacian_edge(self.image)
        self.add_to_history("Laplacian Edge Detection")
        self.display_image_on_canvas()
        self.update_status("Laplacian edge detection applied")

    # ===== AI FEATURES =====

    def ai_color_correction(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        messagebox.showinfo("AI", "AI Color Correction will be implemented!")
        # TODO: Implement AI color correction

    def remove_background(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please open an image first!")
            return
        messagebox.showinfo("AI", "Background Removal will be implemented!")
        # TODO: Implement background removal

    # ===== HELPERS =====

    def undo_action(self):
        messagebox.showinfo("Info", "Undo will be implemented with history system")

    def redo_action(self):
        messagebox.showinfo("Info", "Redo will be implemented with history system")

    def toggle_theme(self):
        """Toggle dark/light theme"""
        current = ctk.get_appearance_mode()
        new_mode = "light" if current == "Dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        self.update_status(f"Theme: {new_mode.capitalize()}")

    def update_status(self, message):
        """Update status bar"""
        self.status_label.configure(text=message)

    # ===== HISTORY SYSTEM =====

    def add_to_history(self, description="Action"):
        """Add current state to history"""
        if not self.history_enabled or self.image is None:
            return

        try:
            # Remove any states after current position (branching)
            if self.history_index < len(self.history) - 1:
                self.history = self.history[: self.history_index + 1]

            # Create history state
            state = {
                "image": self.image.copy(),
                "original": (
                    self.original_image.copy()
                    if self.original_image is not None
                    else None
                ),
                "description": description,
                "timestamp": time.time(),
            }

            # Add to history
            self.history.append(state)

            # Limit history size (remove oldest if needed)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            else:
                self.history_index += 1

            # Update status with history position
            self.update_history_status()

        except Exception as e:
            print(f"⚠️ History error: {e}")

    def undo_action(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_history_state(self.history_index)

            state = self.history[self.history_index]
            self.update_status(
                f"↶ Undo: {state['description']} | {self.get_history_position()}"
            )
        else:
            self.update_status("Nothing to undo")
            messagebox.showinfo("Undo", "No more actions to undo!")

    def redo_action(self):
        """Redo previously undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_history_state(self.history_index)

            state = self.history[self.history_index]
            self.update_status(
                f"↷ Redo: {state['description']} | {self.get_history_position()}"
            )
        else:
            self.update_status("Nothing to redo")
            messagebox.showinfo("Redo", "No more actions to redo!")

    def restore_history_state(self, index):
        """Restore image from history at given index"""
        if 0 <= index < len(self.history):
            state = self.history[index]

            # Temporarily disable history to avoid recursive adds
            self.history_enabled = False

            # Restore images
            self.image = state["image"].copy()
            if state["original"] is not None:
                self.original_image = state["original"].copy()

            # Update display
            self.display_image_on_canvas()

            # Re-enable history
            self.history_enabled = True

    def clear_history(self):
        """Clear all history (use when opening new image)"""
        self.history = []
        self.history_index = -1
        self.update_status("History cleared")

    def get_history_position(self):
        """Get current history position as string"""
        if len(self.history) == 0:
            return "No history"
        return f"Step {self.history_index + 1}/{len(self.history)}"

    def update_history_status(self):
        """Update status bar with history info"""
        if len(self.history) > 0:
            state = self.history[self.history_index]
            self.update_status(
                f"✅ {state['description']} | {self.get_history_position()}"
            )

    def darken_color(self, hex_color):
        """Helper to darken hex color for hover effect"""
        # Simple darkening by reducing RGB values
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(max(0, c - 30) for c in rgb)
        return f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"


# ===== MAIN =====
if __name__ == "__main__":
    app = ModernImageEditor()
    app.mainloop()
