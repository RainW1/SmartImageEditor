import customtkinter as ctk

# Test basic window
app = ctk.CTk()
app.title("Test CustomTkinter")
app.geometry("600x400")

# Set theme
ctk.set_appearance_mode("dark")  # "dark" atau "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Test button
button = ctk.CTkButton(app, text="It Works!", height=50)
button.pack(pady=20, padx=20)

app.mainloop()