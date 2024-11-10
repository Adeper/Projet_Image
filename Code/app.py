import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import skimage as ski
from PIL import Image, ImageTk
import os

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Débruitage d'images")
        self.geometry("1100x720")
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

        # Canvas pour l'image débruitée
        self.canvas_image_debruitee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Ajout d'un menu déroulant pour le choix du mode de détection
        self.mode_debruitage_var = tk.StringVar()
        self.mode_debruitage_var.set("Choisir le mode de débruitage")
        self.modes_debruitage = ["Filtre médian", "Filtre moyenneur", "Filtre bilatéral", "Filtre de Wiener", "Variation totale", "BM3D"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_debruitage_var, values=self.modes_debruitage, command=self.mode_selectionne)
        self.menu_mode_detection.pack(pady=(10, 10))

        self.btn_debruitage = ctk.CTkButton(self.buttons_frame, text="Débruiter l'image", command=self.debruiter, hover_color="darkgrey")

    def mode_selectionne(self, mode):
        self.canvas_image.pack_forget()
        self.canvas_image_debruitee.pack_forget()
        self.label_psnr.pack_forget()
        self.btn_debruitage.pack_forget()

        # A spécialiser selon les modes
        self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
        self.canvas_image.create_text(200, 20, text="Image bruitée", font=("Arial", 12), fill="white")
        self.canvas_image_debruitee.pack(side='left', padx=(20, 20), pady=(20, 20))
        self.canvas_image_debruitee.create_text(200, 20, text="Image débruitée", font=("Arial", 12), fill="white")
        self.label_psnr.pack(pady=(5,5))

    def choisir_image(self):
        dossier_data = os.path.join(os.path.dirname(__file__),"../Data")
        self.image_path = filedialog.askopenfilename(initialdir=dossier_data)
        self.afficher_image(self.image_path, self.canvas_image)

    def afficher_image(self, path, canvas):
        image = Image.open(path)
        image.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        image_tk = ImageTk.PhotoImage(image)
        x = (canvas.winfo_width() - image_tk.width()) / 2
        y = (canvas.winfo_height() - image_tk.height()) / 2
        canvas.delete("all")
        canvas.create_image(x, y, anchor='nw', image=image_tk)
        canvas.image_tk = image_tk

    def debruiter():
        pass

if __name__ == "__main__":
    app = Application()
    app.mainloop()