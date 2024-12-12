import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import skimage as ski
from PIL import Image, ImageTk
import numpy as np
import os
from utils import noise_image_pil, psnr, ssim_score
from methods import median_denoise, mean_denoise, gaussian_denoise, total_variation_denoise, bilateral_denoise, wiener_denoise, fourier_denoise, haar_denoise, bm3d_denoise, CGNet, CGNetDenoise

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Denoisy")
        self.geometry("1920x1080")
        ctk.set_appearance_mode("Dark")
        self.image_path = None
        self.image = None
        self.image_bruitee = None
        self.image_debruitee = None
        #self.CGNetGAN = CGNetDenoise("_CGNet_BSD500/generator.pth")
        self.nima_model = None

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
        self.label_ssim_bruitee = ctk.CTkLabel(self.metrics_frame, text="SSIM entre l'image originale et bruitée : N/A")
        self.label_ssim_bruitee.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.label_ssim_bruitee.grid_remove()
        self.label_ssim_debruitee = ctk.CTkLabel(self.metrics_frame, text="SSIM entre l'image originale et débruitée : N/A")
        self.label_ssim_debruitee.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.label_ssim_debruitee.grid_remove()

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self.buttons_frame, text="Choisir une image", command=self.choisir_image, hover_color="darkgrey")
        self.btn_choisir_image.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # Checkbox pour indiquer si l'image est déjà bruitée
        self.image_bruitee_var = tk.BooleanVar(value=False)
        self.checkbox_image_bruitee = ctk.CTkCheckBox(
            self.buttons_frame,
            text="L'image est déjà bruitée",
            variable=self.image_bruitee_var,
            command=self.on_checkbox_image_bruitee_change
        )
        self.checkbox_image_bruitee.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

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
        self.modes_debruitage = ["Filtre médian", "Filtre moyenneur", "Filtre gaussien", "Filtre bilatéral", "Filtre de Wiener", "Variation totale", "Transformée de Fourier", "Ondelettes de Haar", "BM3D", "CGNet GAN"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_debruitage_var, values=self.modes_debruitage, command=self.mode_selectionne)
        self.menu_mode_detection.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')   

        # Paramètres filtres médian, moyenneur, gaussien et de Wiener
        self.window_size_menu = ctk.CTkOptionMenu(
            self.buttons_frame,
            values=["3", "5", "7", "9"],
            command=self.update_window_size
        )
        self.window_size_label = ctk.CTkLabel(self.buttons_frame, text="Taille de la fenêtre glissante : 3")
        self.window_size_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.window_size_label.grid_remove()
        self.window_size_menu.set("3")
        self.window_size_menu.grid(row=4, column=0, padx=5, pady=5, sticky='nsew')
        self.window_size_menu.grid_remove()

        # Paramètres filtre bilatéral
        self.sigma_color = tk.DoubleVar(value=0.1)
        self.sigma_color_label = ctk.CTkLabel(self.buttons_frame, text="Sigma color : 0.1")
        self.sigma_color_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_color_label.grid_remove()
        self.sigma_color_slider = ctk.CTkSlider(self.buttons_frame, from_=0.01, to=1, command=self.update_sigma_color, state="normal")
        self.sigma_color_slider.set(0.1)
        self.sigma_color_slider.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_color_slider.grid_remove()

        self.sigma_spatial = tk.IntVar(value=15)
        self.sigma_spatial_label = ctk.CTkLabel(self.buttons_frame, text="Sigma spatial : 15")
        self.sigma_spatial_label.grid(row=5, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_spatial_label.grid_remove()
        self.sigma_spatial_slider = ctk.CTkSlider(self.buttons_frame, from_=1, to=20, command=self.update_sigma_spatial, state="normal")
        self.sigma_spatial_slider.set(15)
        self.sigma_spatial_slider.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_spatial_slider.grid_remove()

        # Paramètres variation totale
        self.weight_tv = tk.DoubleVar(value=0.1)
        self.weight_tv_label = ctk.CTkLabel(self.buttons_frame, text="Poids de la régularisation : 0.1")
        self.weight_tv_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.weight_tv_label.grid_remove()
        self.weight_tv_slider = ctk.CTkSlider(self.buttons_frame, from_=0.01, to=0.5, command=self.update_weight_tv, state="normal")
        self.weight_tv_slider.set(0.1)
        self.weight_tv_slider.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.weight_tv_slider.grid_remove()

        # Paramètres Fourier
        self.fourier_threshold = tk.IntVar(value=30)
        self.fourier_threshold_label = ctk.CTkLabel(self.buttons_frame, text="Seuil : 30")
        self.fourier_threshold_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.fourier_threshold_label.grid_remove()
        self.fourier_threshold_slider = ctk.CTkSlider(self.buttons_frame, from_=1, to=100, command=self.update_fourier_threshold, state="normal")
        self.fourier_threshold_slider.set(0.1)
        self.fourier_threshold_slider.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.fourier_threshold_slider.grid_remove()

        # Paramètres Haar
        self.haar_threshold = tk.DoubleVar(value=0.1)
        self.haar_threshold_label = ctk.CTkLabel(self.buttons_frame, text="Seuil : 0.1")
        self.haar_threshold_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.haar_threshold_label.grid_remove()
        self.haar_threshold_slider = ctk.CTkSlider(self.buttons_frame, from_=0.01, to=1, command=self.update_haar_threshold, state="normal")
        self.haar_threshold_slider.set(0.1)
        self.haar_threshold_slider.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.haar_threshold_slider.grid_remove()

        # Paramètres BM3D
        self.sigma_psd = tk.DoubleVar(value=25)
        self.sigma_psd_label = ctk.CTkLabel(self.buttons_frame, text="Sigma PSD : 25")
        self.sigma_psd_label.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_psd_label.grid_remove()
        self.sigma_psd_slider = ctk.CTkSlider(self.buttons_frame, from_=10, to=50, command=self.update_sigma_psd, state="normal")
        self.sigma_psd_slider.set(25)
        self.sigma_psd_slider.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.sigma_psd_slider.grid_remove()

        # Sélection du type de bruit
        self.bruit_type_var = tk.StringVar(value="Type de bruit")
        self.menu_bruit = ctk.CTkOptionMenu(
            self.buttons_frame,
            variable=self.bruit_type_var,
            values=["Gaussien", "Sel et poivre"],
            command=self.on_bruit_type_change
        )
        self.menu_bruit.grid(row=7, column=0, padx=5, pady=5, sticky='nsew')

        # Slider pour la force du bruit
        self.force_bruit = tk.DoubleVar(value=0.05)
        self.label_force_bruit = ctk.CTkLabel(self.buttons_frame, text="Force du bruit : 0.05")
        self.label_force_bruit.grid(row=8, column=0, padx=5, pady=5, sticky='nsew')
        self.label_force_bruit.grid_remove()
        self.slider_force_bruit = ctk.CTkSlider(self.buttons_frame, from_=0.01, to=1, command=self.update_force_bruit, state="disabled")
        self.slider_force_bruit.set(0.05)
        self.slider_force_bruit.grid(row=9, column=0, padx=5, pady=5, sticky='nsew')
        self.slider_force_bruit.grid_remove()

        self.btn_debruitage = ctk.CTkButton(self.buttons_frame, text="Débruiter l'image", command=self.debruiter, hover_color="darkgrey")
        self.btn_debruitage.grid(row=10, column=0, padx=5, pady=5, sticky='nsew')

    def mode_selectionne(self, mode):
        self.window_size_label.grid_remove()
        self.window_size_menu.grid_remove()
        self.sigma_color_label.grid_remove()
        self.sigma_color_slider.grid_remove()
        self.sigma_spatial_label.grid_remove()
        self.sigma_spatial_slider.grid_remove()
        self.weight_tv_label.grid_remove()
        self.weight_tv_slider.grid_remove()
        self.fourier_threshold_label.grid_remove()
        self.fourier_threshold_slider.grid_remove()
        self.haar_threshold_label.grid_remove()
        self.haar_threshold_slider.grid_remove()
        self.sigma_psd_label.grid_remove()
        self.sigma_psd_slider.grid_remove()
        match mode:
            case "Filtre médian":
                self.window_size_label.grid()
                self.window_size_menu.grid()
            case "Filtre moyenneur":
                self.window_size_label.grid()
                self.window_size_menu.grid()
            case "Filtre gaussien":
                self.window_size_label.grid()
                self.window_size_menu.grid()
            case "Filtre bilatéral":
                self.sigma_color_label.grid()
                self.sigma_color_slider.grid()
                self.sigma_spatial_label.grid()
                self.sigma_spatial_slider.grid()
            case "Filtre de Wiener":
                self.window_size_label.grid()
                self.window_size_menu.grid()
            case "Variation totale":
                self.weight_tv_label.grid()
                self.weight_tv_slider.grid()
            case "Transformée de Fourier":
                self.fourier_threshold_label.grid()
                self.fourier_threshold_slider.grid()
            case "Ondelettes de Haar":
                self.haar_threshold_label.grid()
                self.haar_threshold_slider.grid()
            case "BM3D":
                self.sigma_psd_label.grid()
                self.sigma_psd_slider.grid()
            case "CGNet GAN":
                pass

    def choisir_image(self):
        dossier_data = os.path.join(os.path.dirname(__file__),"../Data")
        self.image_path = filedialog.askopenfilename(initialdir=dossier_data)
        self.image = Image.open(self.image_path)
        self.afficher_image(self.image_path, self.canvas_image)
        self.canvas_image_bruitee.grid_remove()
        self.image_bruitee = None
        self.canvas_image_debruitee.grid_remove()
        self.image_debruitee = None
        self.label_psnr_bruitee.grid_remove()
        self.label_psnr_debruitee.grid_remove()
        self.label_ssim_bruitee.grid_remove()
        self.label_ssim_debruitee.grid_remove()

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

    def on_checkbox_image_bruitee_change(self):
        if self.image_bruitee_var.get():
            self.label_force_bruit.grid_remove()
            self.slider_force_bruit.grid_remove()
            self.menu_bruit.grid_remove()
        else:
            self.menu_bruit.grid()
    
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
        self.window_size_label.configure(text=f"Taille de la fenêtre glissante : {value}")

    def update_sigma_color(self, value):
        new_value = round(value, 2)
        self.sigma_color.set(new_value)
        self.sigma_color_label.configure(text=f"Sigma color : {new_value}")

    def update_sigma_spatial(self, value):
        self.sigma_spatial.set(int(value))
        self.sigma_spatial_label.configure(text=f"Sigma spatial : {int(value)}")

    def update_weight_tv(self, value):
        new_value = round(value, 2)
        self.weight_tv.set(new_value)
        self.weight_tv_label.configure(text=f"Poids de la régularisation : {new_value}")

    def update_fourier_threshold(self, value):
        self.fourier_threshold.set(int(value))
        self.fourier_threshold_label.configure(text=f"Seuil : {int(value)}")

    def update_haar_threshold(self, value):
        new_value = round(value, 2)
        self.haar_threshold.set(new_value)
        self.haar_threshold_label.configure(text=f"Seuil : {new_value}")

    def update_sigma_psd(self, value):
        new_value = round(value, 2)
        self.sigma_psd.set(new_value)
        self.sigma_psd_label.configure(text=f"Sigma PSD : {new_value}")
    
    def bruiter_image(self):
        if self.image_bruitee_var.get():
            return
        if self.image:
            noise_type = self.bruit_type_var.get()
            if noise_type not in ["Gaussien", "Sel et poivre"]:
                return
            noise_mapping = {
                "Gaussien": "gaussian",
                "Sel et poivre": "s&p"
            }
            new_noise_type = noise_mapping.get(noise_type)
            
            if new_noise_type:
                self.image_bruitee = noise_image_pil(self.image, self.force_bruit.get(), noise_type=new_noise_type)
                self.image_bruitee.save("../Results/bruitee.png")
                self.canvas_image_bruitee.grid()
                self.afficher_image(self.image_bruitee, self.canvas_image_bruitee)
                canvas_image_bruitee_center_x = self.canvas_image_bruitee.winfo_width() / 2
                self.label_psnr_bruitee.configure(text=f"PSNR entre l'image originale et bruitée : {round(psnr(self.image, self.image_bruitee),2)} dB")
                self.label_psnr_bruitee.grid()
                self.label_ssim_bruitee.configure(text=f"SSIM entre l'image originale et bruitée : {round(ssim_score(self.image, self.image_bruitee),2)}")
                self.label_ssim_bruitee.grid()
                self.canvas_image_bruitee.create_text(canvas_image_bruitee_center_x, 20, text=f"Image avec bruit {noise_type.lower()}", font=("Arial", 12), fill="white", anchor="center")
            else:
                print("Aucun type de bruit valide sélectionné.")
    
    def debruiter(self):
        if self.image_bruitee:
            match self.mode_debruitage_var.get():
                case "Filtre médian":
                    self.image_debruitee = median_denoise(self.image_bruitee, int(self.window_size_menu.get()))
                case "Filtre moyenneur":
                    self.image_debruitee = mean_denoise(self.image_bruitee, int(self.window_size_menu.get()))
                case "Filtre gaussien":
                    self.image_debruitee = gaussian_denoise(self.image_bruitee, int(self.window_size_menu.get()))
                case "Filtre bilatéral":
                    self.image_debruitee = bilateral_denoise(self.image_bruitee, self.sigma_color.get(), self.sigma_spatial.get())
                case "Filtre de Wiener":
                    self.image_debruitee = wiener_denoise(self.image_bruitee, int(self.window_size_menu.get()))
                case "Variation totale":
                    self.image_debruitee = total_variation_denoise(self.image_bruitee, self.weight_tv.get())
                case "Transformée de Fourier":
                    self.image_debruitee = fourier_denoise(self.image_bruitee, self.fourier_threshold.get())
                case "Ondelettes de Haar":
                    self.image_debruitee = haar_denoise(self.image_bruitee, self.haar_threshold.get())
                case "BM3D":
                    self.image_debruitee = bm3d_denoise(self.image_bruitee, self.sigma_psd.get())
                case "CGNet GAN":
                    self.image_debruitee = self.CGNetGAN.denoise(self.image_bruitee)


        elif self.image_bruitee_var.get():
            match self.mode_debruitage_var.get():
                case "Filtre médian":
                    self.image_debruitee = median_denoise(self.image, int(self.window_size_menu.get()))
                case "Filtre moyenneur":
                    self.image_debruitee = mean_denoise(self.image, int(self.window_size_menu.get()))
                case "Filtre gaussien":
                    self.image_debruitee = gaussian_denoise(self.image, int(self.window_size_menu.get()))
                case "Filtre bilatéral":
                    self.image_debruitee = bilateral_denoise(self.image, self.sigma_color.get(), self.sigma_spatial.get())
                case "Filtre de Wiener":
                    self.image_debruitee = wiener_denoise(self.image, int(self.window_size_menu.get()))
                case "Variation totale":
                    self.image_debruitee = total_variation_denoise(self.image, self.weight_tv.get())
                case "Transformée de Fourier":
                    self.image_debruitee = fourier_denoise(self.image, self.fourier_threshold.get())
                case "Ondelettes de Haar":
                    self.image_debruitee = haar_denoise(self.image, self.haar_threshold.get())
                case "BM3D":
                    self.image_debruitee = bm3d_denoise(self.image, self.sigma_psd.get())
                case "CGNet GAN":
                    self.image_debruitee = self.CGNetGAN.denoise(self.image)
        
        if self.image_debruitee is not None:
            self.afficher_image(self.image_debruitee, self.canvas_image_debruitee)
            image_debruitee_pil = self.check_and_convert_image(self.image_debruitee)
            image_debruitee_pil.save("../Results/debruitee.png")
            self.label_psnr_debruitee.configure(text=f"PSNR entre l'image originale et débruitée : {round(psnr(self.image, self.image_debruitee),2)} dB")
            self.label_psnr_debruitee.grid()
            self.label_ssim_debruitee.configure(text=f"SSIM entre l'image originale et débruitée : {round(ssim_score(self.image, self.image_debruitee),2)}")
            self.label_ssim_debruitee.grid()

if __name__ == "__main__":
    app = Application()
    app.mainloop()