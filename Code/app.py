import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import skimage as ski
from PIL import Image, ImageTk
import os
import numpy as np

def noise_image(image, noise_type='gaussian'):
    image_array = np.array(image)

    image_array = image_array / 255.0

    match noise_type:
        case 'gaussian':
            image_noised = ski.util.random_noise(image_array, mode='gaussian', mean=0, var=0.01)
        case 's&p':
            image_noised = ski.util.random_noise(image_array, mode='s&p', amount=0.05)
        case _:
            image_noised = image_array

    image_noised = (image_noised * 255).astype(np.uint8)

    return Image.fromarray(image_noised)

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Débruitage d'images")
        self.geometry("1550x720")
        ctk.set_appearance_mode("Dark")
        self.image_path = None

        # Conteneur global à gauche pour les boutons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='both')

        # Conteneur global pour les métriques
        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.pack(side='bottom', padx=(20, 20), pady=(20, 20), fill='both')

        # Labels pour les métriques
        self.label_psnr = ctk.CTkLabel(self.metrics_frame, text="PSNR : N/A dB")

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self.buttons_frame, text="Choisir une image", command=self.choisir_image, hover_color="darkgrey")
        self.btn_choisir_image.pack(pady=(10, 10))

        # Conteneur global à droite pour les canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='both', expand=True)

        # Canvas pour l'image originale
        self.canvas_image = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Canvas pour l'image bruitée
        self.canvas_image_bruitee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Canvas pour l'image débruitée
        self.canvas_image_debruitee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Ajout d'un menu déroulant pour le choix du mode de détection
        self.mode_debruitage_var = tk.StringVar()
        self.mode_debruitage_var.set("Choisir le mode de débruitage")
        self.modes_debruitage = ["Filtre médian", "Filtre moyenneur", "Filtre bilatéral", "Filtre de Wiener", "Variation totale", "BM3D"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_debruitage_var, values=self.modes_debruitage, command=self.mode_selectionne)
        self.menu_mode_detection.pack(pady=(10, 10))

        self.btn_debruitage = ctk.CTkButton(self.buttons_frame, text="Débruiter l'image", command=self.debruiter, hover_color="darkgrey")
        
        # Sélection du type de bruit
        self.bruit_type_var = tk.StringVar(value="Type de bruit")
        self.menu_bruit = ctk.CTkOptionMenu(
            self.buttons_frame,
            variable=self.bruit_type_var,
            values=["Gaussien", "Sel et poivre"],
            command=self.on_noise_type_change
        )
        self.menu_bruit.pack(pady=(10, 10))
        
        # Bouton pour bruiter l'image
        self.btn_bruiter_image = ctk.CTkButton(self.buttons_frame, text="Bruiter l'image", command=self.bruiter_image, hover_color="darkgrey")
        self.btn_bruiter_image.pack(pady=(10, 10))

        # Initialiser les canvas : afficher l'image originale et débruitée
        self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
        self.canvas_image.create_text(200, 20, text="Image originale", font=("Arial", 12), fill="white")

        self.canvas_image_debruitee.pack(side='left', padx=(20, 20), pady=(20, 20))
        self.canvas_image_debruitee.create_text(200, 20, text="Image débruitée", font=("Arial", 12), fill="white")

        # Cacher l'image bruitée au début
        self.canvas_image_bruitee.pack_forget()

    def mode_selectionne(self, mode):
        pass  # A vous de personnaliser cette fonction selon vos besoins

    def choisir_image(self):
        dossier_data = os.path.join(os.path.dirname(__file__),"../Data")
        self.image_path = filedialog.askopenfilename(initialdir=dossier_data)
        self.afficher_image(self.image_path, self.canvas_image)

    def afficher_image(self, image_or_path, canvas):
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        else:
            image = image_or_path
        image.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        image_tk = ImageTk.PhotoImage(image)
        x = (canvas.winfo_width() - image_tk.width()) / 2
        y = (canvas.winfo_height() - image_tk.height()) / 2
        canvas.delete("all")
        canvas.create_image(x, y, anchor='nw', image=image_tk)
        canvas.image_tk = image_tk

    def on_noise_type_change(self, noise_type):
        print(f"Type de bruit sélectionné : {noise_type}")

    def bruiter_image(self):
        if self.image_path:
            # Charger l'image sélectionnée
            image = Image.open(self.image_path)
            
            # Récupérer le type de bruit choisi
            noise_type = self.bruit_type_var.get()
            print(f"Type de bruit sélectionné : {noise_type}")
            
            # Utiliser un dictionnaire pour mapper les types de bruit
            noise_mapping = {
                "Gaussien": "gaussian",
                "Sel et poivre": "s&p"
            }
            
            # Obtenir le type de bruit à appliquer
            noise_type = noise_mapping.get(noise_type)
            
            if noise_type:
                # Appliquer le bruit
                image_bruitee = noise_image(image, noise_type=noise_type)
                
                # Afficher l'image bruitée dans le canvas bruité
                self.afficher_image(image_bruitee, self.canvas_image_bruitee)
                self.canvas_image_bruitee.create_text(200, 20, text=f"Image avec bruit {noise_type}", font=("Arial", 12), fill="white")

                # Afficher le canvas de l'image bruitée
                self.canvas_image_bruitee.pack(side='left', padx=(20, 20), pady=(20, 20))
            else:
                print("Aucun type de bruit valide sélectionné.")
    
    def debruiter(self):
        pass  # Vous pouvez ajouter votre logique de débruitage ici

if __name__ == "__main__":
    app = Application()
    app.mainloop()