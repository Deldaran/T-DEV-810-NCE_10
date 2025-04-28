import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms, models
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
matplotlib.use("TkAgg")


class PneumoniaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de Pneumonie")
        self.root.geometry("900x700")
        
        # Définir le device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chargement du modèle
        self.load_model()
        
        # Variables pour les résultats
        self.normal_probas = []
        self.pneumonia_probas = []
        
        # Création de l'interface
        self.create_widgets()
        
    def load_model(self):
        try:
            # Définir l'architecture du modèle (MobileNetV2)
            import torch.nn as nn
            
            # Utiliser weights=None au lieu de pretrained=False (pour éliminer l'avertissement)
            self.model = models.efficientnet_b2(weights=None, num_classes=1)
            
            # Adapter la première couche pour accepter 1 canal (niveaux de gris)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            
            # Remplacer le classifier pour la classification binaire
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, 1)
            )
            
            # Charger les poids du modèle (avec weights_only=True pour éviter l'avertissement de sécurité)
            self.model.load_state_dict(torch.load('models/final_model.pth', map_location=self.device, weights_only=True))
            
            # Déplacer le modèle vers le device approprié
            self.model = self.model.to(self.device)
            
            # Mettre en mode évaluation
            self.model.eval()
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
        
        # Frame pour les boutons
        button_frame = Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Bouton pour charger une image unique
        self.load_button = Button(button_frame, text="Charger une image", command=self.load_image, font=("Arial", 12))
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        # Bouton pour analyser un dossier
        self.folder_button = Button(button_frame, text="Analyser un dossier", command=self.load_folder, font=("Arial", 12))
        self.folder_button.pack(side=tk.LEFT, padx=10)
        
        # Notebook pour organiser les onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Onglet pour l'analyse d'image unique
        self.single_frame = Frame(self.notebook)
        self.notebook.add(self.single_frame, text="Image unique")
        
        # Onglet pour l'analyse par lot
        self.batch_frame = Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Analyse par lot")
        
        # Ajout d'un onglet pour les métriques avancées
        self.metrics_frame = Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Métriques avancées")
        
        # Contenu de l'onglet Image unique
        # Frame pour l'image
        self.image_frame = Frame(self.single_frame, width=400, height=400, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10)
        
        # Label pour afficher l'image
        self.image_label = Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        # Label pour afficher le résultat
        self.result_frame = Frame(self.single_frame)
        self.result_frame.pack(pady=10, fill=tk.X)
        
        self.result_label = Label(self.result_frame, text="Chargez une image pour obtenir un diagnostic", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        # Label pour afficher la probabilité
        self.probability_label = Label(self.result_frame, text="", font=("Arial", 12))
        self.probability_label.pack(pady=5)
        
        # Contenu de l'onglet Analyse par lot
        # Frame pour le graphique
        self.graph_frame = Frame(self.batch_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
            # Passer à l'onglet Image unique
            self.notebook.select(0)
    
    def load_folder(self):
        folder_path = filedialog.askdirectory(
            title="Sélectionner un dossier"
        )
        
        if folder_path:
            self.status_bar.config(text=f"Dossier sélectionné: {os.path.basename(folder_path)}")
            self.process_folder(folder_path)
            # Passer à l'onglet Analyse par lot
            self.notebook.select(1)
    
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
            
            # Définir les transformations
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])  # Normalisation ImageNet
            ])
            
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)  # Ajouter dimension batch et envoyer au device
            
            if self.model:
                # Désactiver le calcul du gradient pour l'inférence
                with torch.no_grad():
                    # Faire la prédiction
                    output = self.model(img_tensor)
                    prediction = torch.sigmoid(output).item()  # Appliquer la sigmoïde pour obtenir une probabilité
                
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
    
    def process_folder(self, folder_path):
        # Réinitialiser les listes de résultats
        self.normal_probas = []
        self.pneumonia_probas = []
        
        # Pour stocker les prédictions et les vrais labels
        self.true_labels = []  # 0 pour normal, 1 pour pneumonie
        self.predicted_probas = []
        
        try:
            # Vérifier la structure du dossier
            normal_dir = os.path.join(folder_path, "NORMAL")
            pneumonia_dir = os.path.join(folder_path, "PNEUMONIA")
            
            # Traiter les images NORMAL si le dossier existe
            if os.path.exists(normal_dir):
                self.status_bar.config(text="Traitement des images NORMAL...")
                self.root.update()  # Forcer la mise à jour de l'interface
                self.normal_probas = self.process_images_in_directory(normal_dir)
                
                # Ajouter aux données pour les métriques
                self.true_labels.extend([0] * len(self.normal_probas))
                self.predicted_probas.extend(self.normal_probas)
            
            # Traiter les images PNEUMONIA si le dossier existe
            if os.path.exists(pneumonia_dir):
                self.status_bar.config(text="Traitement des images PNEUMONIA...")
                self.root.update()  # Forcer la mise à jour de l'interface
                self.pneumonia_probas = self.process_images_in_directory(pneumonia_dir)
                
                # Ajouter aux données pour les métriques
                self.true_labels.extend([1] * len(self.pneumonia_probas))
                self.predicted_probas.extend(self.pneumonia_probas)
            
            # Créer et afficher la courbe d'histogramme
            self.create_plot()
            
            # Créer et afficher les métriques avancées
            self.create_advanced_metrics()
            
            self.status_bar.config(text=f"Analyse terminée: {len(self.normal_probas)} images NORMAL, {len(self.pneumonia_probas)} images PNEUMONIA")
        
        except Exception as e:
            self.status_bar.config(text=f"Erreur lors de l'analyse du dossier: {str(e)}")
            print(f"Exception détaillée: {e}")  # Pour le débogage
    
    def process_images_in_directory(self, directory):
        probabilities = []
        extensions = ('.jpg', '.jpeg', '.png')
        
        # Parcourir tous les fichiers du dossier
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                file_path = os.path.join(directory, filename)
                
                try:
                    # Prétraitement de l'image
                    img = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
                    
                    # Définir les transformations
                    preprocess = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalisation ImageNet
                    ])
                    
                    img_tensor = preprocess(img).unsqueeze(0).to(self.device)
                    
                    # Prédiction
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        prediction = torch.sigmoid(output).item()
                    
                    probabilities.append(prediction)
                
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {file_path}: {str(e)}")
        
        return probabilities
    
    def create_plot(self):
        # Supprimer le graphique précédent s'il existe
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Vérifier si nous avons des données à afficher
        if not self.normal_probas and not self.pneumonia_probas:
            no_data_label = Label(self.graph_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
        
        # Créer une figure matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Créer l'histogramme des probabilités
        bins = np.linspace(0, 1, 21)  # 20 intervalles entre 0 et 1
        
        # Toujours créer les histogrammes même s'ils sont vides
        ax.hist(self.normal_probas if self.normal_probas else [0], bins=bins, alpha=0.5, label='NORMAL', color='green')
        ax.hist(self.pneumonia_probas if self.pneumonia_probas else [0], bins=bins, alpha=0.5, label='PNEUMONIE', color='red')
        
        # S'assurer que les étiquettes sont définies même si les listes sont vides
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # Si aucune légende n'est générée, créer des entrées fictives pour la légende
            import matplotlib.patches as mpatches
            normal_patch = mpatches.Patch(color='green', alpha=0.5, label='NORMAL')
            pneumo_patch = mpatches.Patch(color='red', alpha=0.5, label='PNEUMONIE')
            handles = [normal_patch, pneumo_patch]
            labels = ['NORMAL', 'PNEUMONIE']
        
        # Ajouter les légendes manuellement
        ax.legend(handles, labels)
        
        # Ajouter les titres et grille
        ax.set_title('Distribution des probabilités de pneumonie')
        ax.set_xlabel('Probabilité de pneumonie')
        ax.set_ylabel('Nombre d\'images')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une ligne verticale à 0.5 (seuil de décision)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
        
        # Intégrer le graphique dans l'interface Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Ajouter des statistiques sous le graphique
        stats_frame = Frame(self.graph_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculer quelques statistiques
        normal_mean = np.mean(self.normal_probas) if self.normal_probas else 0
        pneumonia_mean = np.mean(self.pneumonia_probas) if self.pneumonia_probas else 0
        
        normal_correct = sum(p <= 0.5 for p in self.normal_probas) if self.normal_probas else 0
        pneumonia_correct = sum(p > 0.5 for p in self.pneumonia_probas) if self.pneumonia_probas else 0
        
        normal_accuracy = normal_correct / len(self.normal_probas) * 100 if self.normal_probas else 0
        pneumonia_accuracy = pneumonia_correct / len(self.pneumonia_probas) * 100 if self.pneumonia_probas else 0
        
        # Calculer les taux d'erreur
        normal_error = 100 - normal_accuracy
        pneumonia_error = 100 - pneumonia_accuracy
        
        overall_correct = normal_correct + pneumonia_correct
        overall_total = len(self.normal_probas) + len(self.pneumonia_probas)
        overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
        overall_error = 100 - overall_accuracy
        
        # Afficher les statistiques avec les taux d'erreur
        stats_text = f"Précision NORMAL: {normal_accuracy:.2f}% (Erreur: {normal_error:.2f}%) | Précision PNEUMONIE: {pneumonia_accuracy:.2f}% (Erreur: {pneumonia_error:.2f}%) | Précision globale: {overall_accuracy:.2f}% (Erreur: {overall_error:.2f}%)"
        stats_label = Label(stats_frame, text=stats_text, font=("Arial", 10))
        stats_label.pack(pady=5)
    
    def create_advanced_metrics(self):
        # Supprimer les graphiques précédents s'ils existent
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Vérifier si on a des données
        if not self.true_labels or not self.predicted_probas:
            no_data_label = Label(self.metrics_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
        
        # Convertir les listes en tableaux numpy pour faciliter le traitement
        true_labels = np.array(self.true_labels)
        predicted_probas = np.array(self.predicted_probas)
        
        # Prédictions binaires en utilisant le seuil 0.5
        predicted_labels = (predicted_probas > 0.5).astype(int)
        
        # Créer une figure avec 2x2 sous-plots
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Matrice de confusion
        ax1 = fig.add_subplot(221)
        cm = confusion_matrix(true_labels, predicted_labels)
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Matrice de confusion')
        
        # Ajouter les valeurs dans les cellules
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['NORMAL', 'PNEUMONIE'])
        ax1.set_yticklabels(['NORMAL', 'PNEUMONIE'])
        ax1.set_ylabel('Vrai label')
        ax1.set_xlabel('Prédiction')
        
        # 2. Courbe ROC
        ax2 = fig.add_subplot(222)
        fpr, tpr, _ = roc_curve(true_labels, predicted_probas)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Taux de faux positifs')
        ax2.set_ylabel('Taux de vrais positifs')
        ax2.set_title('Courbe ROC')
        ax2.legend(loc="lower right")
        
        # 3. Courbe Précision-Rappel
        ax3 = fig.add_subplot(223)
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probas)
        
        ax3.plot(recall, precision, color='green', lw=2)
        ax3.set_xlabel('Rappel')
        ax3.set_ylabel('Précision')
        ax3.set_title('Courbe Précision-Rappel')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        
        # 4. Tableau des métriques
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculer les métriques
        TP = cm[1, 1]  # Vrais positifs
        TN = cm[0, 0]  # Vrais négatifs
        FP = cm[0, 1]  # Faux positifs
        FN = cm[1, 0]  # Faux négatifs
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        error_rate = 1 - accuracy
        
        precision_metric = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_metric = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision_metric * recall_metric) / (precision_metric + recall_metric) if (precision_metric + recall_metric) > 0 else 0
        
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Créer un tableau de métriques
        metrics_table = [
            ["Métrique", "Valeur"],
            ["Précision", f"{accuracy:.4f}"],
            ["Taux d'erreur", f"{error_rate:.4f}"],
            ["Précision (Precision)", f"{precision_metric:.4f}"],
            ["Rappel (Recall/Sensibilité)", f"{recall_metric:.4f}"],
            ["Score F1", f"{f1_score:.4f}"],
            ["Spécificité", f"{specificity:.4f}"],
            ["AUC-ROC", f"{roc_auc:.4f}"]
        ]
        
        # Afficher le tableau
        ax4.table(cellText=metrics_table, loc='center', cellLoc='center', colWidths=[0.5, 0.3])
        ax4.set_title('Métriques de performance')
        
        plt.tight_layout()
        
        # Intégrer la figure dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectorApp(root)
    root.mainloop()