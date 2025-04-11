import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

class PneumoniaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de Pneumonie")
        self.root.geometry("800x600")
        
        # Chargement du modèle
        self.load_model()
        
        # Création de l'interface
        self.create_widgets()
        
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model('models/best_model.h5')
            print("Modèle chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            self.model = None
    
    def create_widgets(self):
        # Frame principal
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = Label(main_frame, text="Détecteur de Pneumonie par Rayons X", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Bouton pour charger une image
        self.load_button = Button(main_frame, text="Charger une image", command=self.load_image, font=("Arial", 12))
        self.load_button.pack(pady=10)
        
        # Frame pour l'image
        self.image_frame = Frame(main_frame, width=400, height=400, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10)
        
        # Label pour afficher l'image
        self.image_label = Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        # Label pour afficher le résultat
        self.result_frame = Frame(main_frame)
        self.result_frame.pack(pady=10, fill=tk.X)
        
        self.result_label = Label(self.result_frame, text="Chargez une image pour obtenir un diagnostic", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        # Label pour afficher la probabilité
        self.probability_label = Label(self.result_frame, text="", font=("Arial", 12))
        self.probability_label.pack(pady=5)
        
        # Status bar
        self.status_bar = Label(self.root, text="Prêt", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            self.status_bar.config(text=f"Image chargée: {os.path.basename(file_path)}")
            self.process_image(file_path)
    
    def process_image(self, file_path):
        # Chargement et prétraitement de l'image
        try:
            # Charger l'image pour l'affichage
            display_image = Image.open(file_path)
            display_image = display_image.resize((300, 300), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(display_image)
            
            # Afficher l'image
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Garder une référence
            
            # Prétraitement pour le modèle
            img = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0  # Normalisation
            img_array = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch
            img_array = np.expand_dims(img_array, axis=3)  # Ajouter dimension channel pour grayscale
            
            if self.model:
                # Faire la prédiction
                prediction = self.model.predict(img_array)[0][0]
                
                # Afficher le résultat
                if prediction > 0.5:
                    result_text = "Diagnostic: PNEUMONIE détectée"
                    result_color = "red"
                else:
                    result_text = "Diagnostic: NORMAL (pas de pneumonie)"
                    result_color = "green"
                
                self.result_label.config(text=result_text, fg=result_color)
                self.probability_label.config(text=f"Probabilité de pneumonie: {prediction*100:.2f}%")
            else:
                self.result_label.config(text="Erreur: Modèle non chargé", fg="red")
                self.probability_label.config(text="")
                
        except Exception as e:
            self.status_bar.config(text=f"Erreur: {str(e)}")
            self.result_label.config(text="Erreur lors du traitement de l'image", fg="red")
            self.probability_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectorApp(root)
    root.mainloop()