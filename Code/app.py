import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import skimage as ski
from PIL import Image, ImageTk
import numpy as np
import os
from utils import noise_image_pil, psnr
from methods import median_denoise, gaussian_denoise

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Débruitage d'images")
        self.geometry("1920x1080")
        ctk.set_appearance_mode("Dark")
        self.image_path = None
        self.image = None
        self.image_bruitee = None
        self.image_debruitee = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Colonne gauche pour les boutons
        self.grid_columnconfigure(1, weight=1)  # Colonne droite pour les canvas
        self.grid_rowconfigure(1, weight=0) # Ligne en bas pour les métriques

        # Conteneur global à gauche pour les boutons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Conteneur global pour les métriques
        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        # Labels pour les métriques
        self.label_psnr_bruitee = ctk.CTkLabel(self.metrics_frame, text="PSNR entre l'image originale et bruitée : N/A dB")
        self.label_psnr_bruitee.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.label_psnr_bruitee.grid_remove()
        self.label_psnr_debruitee = ctk.CTkLabel(self.metrics_frame, text="PSNR entre l'image originale et débruitée : N/A dB")
        self.label_psnr_debruitee.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.label_psnr_debruitee.grid_remove()

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self.buttons_frame, text="Choisir une image", command=self.choisir_image, hover_color="darkgrey")
        self.btn_choisir_image.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # Conteneur global à droite pour les canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, padx=10, pady=10)
        self.canvas_frame.grid_remove()
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=1)
        self.canvas_frame.columnconfigure(2, weight=1)

        # Canvas pour l'image originale
        self.canvas_image = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_image.grid(row=1, column=0, padx=10, pady=10)
        self.canvas_image.grid_remove()

        # Canvas pour l'image bruitée
        self.canvas_image_bruitee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_image_bruitee.grid(row=1, column=1, padx=10, pady=10)
        self.canvas_image_bruitee.grid_remove()

        # Canvas pour l'image débruitée
        self.canvas_image_debruitee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_image_debruitee.grid(row=1, column=2, padx=10, pady=10)
        self.canvas_image_debruitee.grid_remove()

        # Ajout d'un menu déroulant pour le choix du mode de détection
        self.mode_debruitage_var = tk.StringVar()
        self.mode_debruitage_var.set("Choisir le mode de débruitage")
        self.modes_debruitage = ["Filtre médian", "Filtre moyenneur", "Variation totale"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_debruitage_var, values=self.modes_debruitage, command=self.mode_selectionne)
        self.menu_mode_detection.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        # Paramètres filtres
        self.window_size_menu = ctk.CTkOptionMenu(
            self.buttons_frame,
            values=["3", "5", "7", "9"],
            command=self.update_window_size
        )
        self.window_size_menu.set("3")
        self.window_size_menu.grid(row=3, column=0, padx=5, pady=5, sticky='nsew')
        self.window_size_menu.grid_remove()

        # Paramètres filtre médian
        self.window_size_label = ctk.CTkLabel(self.buttons_frame, text="Taille de la fenêtre glissante : 3")
        self.window_size_label.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.window_size_label.grid_remove()
        

        # Sélection du type de bruit
        self.bruit_type_var = tk.StringVar(value="Type de bruit")
        self.menu_bruit = ctk.CTkOptionMenu(
            self.buttons_frame,
            variable=self.bruit_type_var,
            values=["Gaussien", "Sel et poivre"],
            command=self.on_bruit_type_change
        )
        self.menu_bruit.grid(row=4, column=0, padx=5, pady=5, sticky='nsew')

        # Slider pour la force du bruit
        self.force_bruit = tk.DoubleVar(value=0.05)
        self.label_force_bruit = ctk.CTkLabel(self.buttons_frame, text="Force du bruit : 0.05")
        self.label_force_bruit.grid(row=5, column=0, padx=5, pady=5, sticky='nsew')
        self.label_force_bruit.grid_remove()
        self.slider_force_bruit = ctk.CTkSlider(self.buttons_frame, from_=0.01, to=1, command=self.update_force_bruit, state="disabled")
        self.slider_force_bruit.set(0.05)
        self.slider_force_bruit.grid(row=6, column=0, padx=5, pady=5, sticky='nsew')
        self.slider_force_bruit.grid_remove()

        self.btn_debruitage = ctk.CTkButton(self.buttons_frame, text="Débruiter l'image", command=self.debruiter, hover_color="darkgrey")
        self.btn_debruitage.grid(row=7, column=0, padx=5, pady=5, sticky='nsew')

    def mode_selectionne(self, mode):
        self.window_size_label.grid_remove()
        self.window_size_menu.grid_remove()
        match mode:
            case "Filtre médian":
                self.window_size_label.grid()
                self.window_size_menu.grid()
            case "Filtre moyenneur":
                self.window_size_label.grid()
                self.window_size_menu.grid()


    def choisir_image(self):
        dossier_data = os.path.join(os.path.dirname(__file__),"../Data")
        self.image_path = filedialog.askopenfilename(initialdir=dossier_data)
        self.image = Image.open(self.image_path)
        self.afficher_image(self.image_path, self.canvas_image)

    def check_and_convert_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        return image

    def afficher_image(self, image_or_path, canvas):
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        else:
            image = self.check_and_convert_image(image_or_path)
        self.canvas_frame.grid()
        canvas.grid()
        canvas.update_idletasks()
        image.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        image_tk = ImageTk.PhotoImage(image)
        x = (canvas.winfo_width() - image_tk.width()) / 2
        y = (canvas.winfo_height() - image_tk.height()) / 2
        canvas.delete("all")
        canvas.create_image(x, y, anchor='nw', image=image_tk)
        canvas.image_tk = image_tk
        match canvas:
            case self.canvas_image:
                canvas_image_center_x = self.canvas_image.winfo_width() / 2
                canvas.create_text(canvas_image_center_x, 20, text="Image originale", font=("Arial", 12), fill="white", anchor="center")
            case self.canvas_image_debruitee:
                canvas_image_debruitee_center_x = self.canvas_image_debruitee.winfo_width() / 2
                canvas.create_text(canvas_image_debruitee_center_x, 20, text=f"Image débruitée par {self.mode_debruitage_var.get().lower()}", font=("Arial", 12), fill="white", anchor="center")

    def on_bruit_type_change(self, value):
        self.label_force_bruit.grid()
        self.slider_force_bruit.grid()
        self.slider_force_bruit.configure(state="normal")

    def update_force_bruit(self, value):
        new_value = round(value, 2)
        self.force_bruit.set(new_value)
        if(self.bruit_type_var != "Type de bruit"):
            self.label_force_bruit.configure(text=f"Force du bruit : {new_value}")
            self.bruiter_image()

    def update_window_size(self, value):
        self.window_size_menu.configure(text=f"Taille de la fenêtre glissante : {value}") 
    
    def bruiter_image(self):
        if self.image:
            noise_type = self.bruit_type_var.get()
            noise_mapping = {
                "Gaussien": "gaussian",
                "Sel et poivre": "s&p"
            }
            new_noise_type = noise_mapping.get(noise_type)
            
            if noise_type:
                self.image_bruitee = noise_image_pil(self.image, self.force_bruit.get(), noise_type=new_noise_type)
                self.canvas_image_bruitee.grid()
                self.afficher_image(self.image_bruitee, self.canvas_image_bruitee)
                canvas_image_bruitee_center_x = self.canvas_image_bruitee.winfo_width() / 2
                self.label_psnr_bruitee.configure(text=f"PSNR entre l'image originale et bruitée : {round(psnr(self.image, self.image_bruitee),2)} dB")
                self.label_psnr_bruitee.grid()
                self.canvas_image_bruitee.create_text(canvas_image_bruitee_center_x, 20, text=f"Image avec bruit {noise_type.lower()}", font=("Arial", 12), fill="white", anchor="center")
            else:
                print("Aucun type de bruit valide sélectionné.")
    
    def debruiter(self):
        if self.image_bruitee:
            match self.mode_debruitage_var.get():
                case "Filtre médian":
                    self.image_debruitee = median_denoise(self.image_bruitee, int(self.window_size_menu.get()))
                    self.afficher_image(self.image_debruitee, self.canvas_image_debruitee)
                case "Filtre moyenneur":
                    self.image_debruitee = gaussian_denoise(self.image_bruitee)
                    self.afficher_image(self.image_debruitee, self.canvas_image_debruitee)
            if self.image_debruitee is not None:
                self.label_psnr_debruitee.configure(text=f"PSNR entre l'image originale et débruitée : {round(psnr(self.image, self.image_debruitee),2)} dB")
                self.label_psnr_debruitee.grid()

if __name__ == "__main__":
    app = Application()
    app.mainloop()