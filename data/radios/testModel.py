import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, ttk, Scale
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms, models
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn

matplotlib.use("TkAgg")

class ResNet34WithDropout(nn.Module):
    def __init__(self, base_model, num_classes, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PneumoniaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de Pneumonie - 3 Classes")
        self.root.geometry("1200x900")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

        # Classes disponibles
        self.classes = ['NORMAL', 'BACTERIA', 'VIRUS']
        self.class_colors = ['blue', 'orange', 'red']
        
        # Variables pour stocker les résultats
        self.probas = []
        self.true_labels = []
        self.predicted_labels = []
        self.file_paths = []
        self.current_image_probs = None

        self.create_widgets()

    def load_model(self):
        try:
            checkpoint = torch.load('models/best_model.pth', map_location=self.device, weights_only=False)
            
            # Print checkpoint structure for diagnosis
            print("Checkpoint keys:", checkpoint.keys())
            
            # Get the state dict from the checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Print first few keys to understand model structure
            print("First 5 state dict keys:", list(state_dict.keys())[:5])
            
            # Créer le modèle pour 3 classes
            base_model = models.resnet34(weights=None)
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = ResNet34WithDropout(base_model, num_classes=3, dropout_p=0.5)
            
            try:
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Failed to load state dict: {e}")
                self.model = None
                return
                
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Modèle chargé avec succès (3 classes)")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            self.model = None

    def create_scrollable_frame(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame

    def create_widgets(self):
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = Label(main_frame, text="Détecteur de Pneumonie - Normal/Bacteria/Virus", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        button_frame = Frame(main_frame)
        button_frame.pack(pady=10)

        self.load_button = Button(button_frame, text="Charger une image", command=self.load_image, font=("Arial", 12))
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.folder_button = Button(button_frame, text="Analyser un dossier", command=self.load_folder, font=("Arial", 12))
        self.folder_button.pack(side=tk.LEFT, padx=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.single_frame = Frame(self.notebook)
        self.notebook.add(self.single_frame, text="Image unique")

        self.batch_frame = Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Analyse par lot")

        self.metrics_frame = Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Métriques avancées")

        self.roc_frame = Frame(self.notebook)
        self.notebook.add(self.roc_frame, text="Courbe ROC")

        # Image unique
        self.image_frame = Frame(self.single_frame, width=400, height=400, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10)
        self.image_label = Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        self.result_frame = Frame(self.single_frame)
        self.result_frame.pack(pady=10, fill=tk.X)
        self.result_label = Label(self.result_frame, text="Chargez une image pour obtenir un diagnostic", font=("Arial", 14))
        self.result_label.pack(pady=10)
        self.probability_label = Label(self.result_frame, text="", font=("Arial", 12))
        self.probability_label.pack(pady=5)

        # Batch frame avec scroll
        self.graph_scroll_frame = Frame(self.batch_frame)
        self.graph_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.graph_frame = self.create_scrollable_frame(self.graph_scroll_frame)

        # Metrics frame avec scroll
        self.metrics_scroll_frame = Frame(self.metrics_frame)
        self.metrics_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.metrics_inner_frame = self.create_scrollable_frame(self.metrics_scroll_frame)

        # ROC frame
        self.roc_scroll_frame = Frame(self.roc_frame)
        self.roc_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.roc_inner_frame = self.create_scrollable_frame(self.roc_scroll_frame)

        # Status bar
        self.status_bar = Label(self.root, text="Prêt", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def get_true_label_from_path(self, file_path):
        """Détermine le label vrai basé sur le chemin du fichier"""
        fname = os.path.basename(file_path).lower()
        parent_folder = os.path.basename(os.path.dirname(file_path)).upper()
        
        # Vérifier d'abord le nom du dossier parent
        if 'NORMAL' in parent_folder:
            return 0
        elif 'BACTERIA' in parent_folder or 'BACTERIAL' in parent_folder:
            return 1
        elif 'VIRUS' in parent_folder or 'VIRAL' in parent_folder:
            return 2
        
        # Vérifier ensuite le nom du fichier
        if 'normal' in fname:
            return 0
        elif 'bacteria' in fname or 'bacterial' in fname:
            return 1
        elif 'virus' in fname or 'viral' in fname:
            return 2
        
        # Par défaut, retourner 0 (normal) si aucune indication claire
        print(f"Attention: impossible de déterminer la classe pour {file_path}, assigné à NORMAL")
        return 0

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.status_bar.config(text=f"Image chargée: {os.path.basename(file_path)}")
            self.process_image(file_path)
            self.notebook.select(0)

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier")
        if folder_path:
            self.status_bar.config(text=f"Dossier sélectionné: {os.path.basename(folder_path)}")
            self.process_folder(folder_path)
            self.notebook.select(1)

    def process_image(self, file_path):
        try:
            # Afficher l'image
            display_image = Image.open(file_path)
            display_image = display_image.resize((300, 300), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            # Traitement de l'image pour le modèle
            img = Image.open(file_path).convert('L')
            preprocess = transforms.Compose([
                transforms.Resize(400, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)

            if self.model:
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                
                # Sauvegarder les probabilités pour l'image actuelle
                self.current_image_probs = probs
                
                # Prédiction basée sur la probabilité maximale
                pred_label = int(np.argmax(probs))
                pred_class = self.classes[pred_label]
                color = self.class_colors[pred_label]
                
                # Affichage des résultats
                prob_text = ", ".join([f"{c}: {p*100:.2f}%" for c, p in zip(self.classes, probs)])
                self.result_label.config(text=f"Diagnostic: {pred_class}", fg=color)
                self.probability_label.config(text=prob_text)
            else:
                self.result_label.config(text="Erreur: Modèle non chargé", fg="red")
                self.probability_label.config(text="")
        except Exception as e:
            self.status_bar.config(text=f"Erreur: {str(e)}")
            self.result_label.config(text="Erreur lors du traitement de l'image", fg="red")
            self.probability_label.config(text="")

    def process_folder(self, folder_path):
        self.probas = []
        self.true_labels = []
        self.predicted_labels = []
        self.file_paths = []
        
        try:
            self.status_bar.config(text="Traitement des images...")
            self.root.update()
            
            for root_dir, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root_dir, file)
                        
                        # Déterminer le label vrai
                        true_label = self.get_true_label_from_path(file_path)
                        
                        # Traiter l'image
                        probs = self.process_single_image_for_batch(file_path)
                        if probs is not None:
                            pred_label = int(np.argmax(probs))
                            
                            self.true_labels.append(true_label)
                            self.predicted_labels.append(pred_label)
                            self.probas.append(probs)
                            self.file_paths.append(file_path)
            
            # Créer les visualisations
            self.create_plot_multiclass()
            self.create_advanced_metrics_multiclass()
            self.create_roc_curve_multiclass()
            
            self.status_bar.config(text=f"Analyse terminée: {len(self.true_labels)} images traitées")
        except Exception as e:
            self.status_bar.config(text=f"Erreur lors de l'analyse du dossier: {str(e)}")
            print(f"Exception détaillée: {e}")

    def process_single_image_for_batch(self, file_path):
        try:
            img = Image.open(file_path).convert('L')
            preprocess = transforms.Compose([
                transforms.Resize(400, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            return probs
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {file_path}: {str(e)}")
            return None

    def create_plot_multiclass(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        if not self.probas:
            no_data_label = Label(self.graph_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return

        probas = np.array(self.probas)
        
        # Créer un graphique avec subplots pour chaque classe
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (cls, color, ax) in enumerate(zip(self.classes, self.class_colors, axes)):
            # Histogramme des probabilités pour chaque classe
            ax.hist(probas[:, i], bins=20, color=color, alpha=0.7, edgecolor='black')
            
            # Ligne de moyenne
            mean_val = np.mean(probas[:, i])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
            ax.text(mean_val + 0.02, ax.get_ylim()[1]*0.8, f"Moy: {mean_val:.3f}", 
                   color='red', fontweight='bold')
            
            ax.set_title(f'Distribution - {cls}')
            ax.set_xlabel('Probabilité')
            ax.set_ylabel('Fréquence')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ajouter le graphique à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistiques
        stats_frame = Frame(self.graph_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        means = [np.mean(probas[:, i]) for i in range(3)]
        stats_text = " | ".join([f"{cls}: {mean:.3f}" for cls, mean in zip(self.classes, means)])
        stats_label = Label(stats_frame, text=f"Moyennes - {stats_text}", font=("Arial", 12))
        stats_label.pack()

    def create_advanced_metrics_multiclass(self):
        for widget in self.metrics_inner_frame.winfo_children():
            widget.destroy()
            
        if not self.true_labels or not self.predicted_labels:
            no_data_label = Label(self.metrics_inner_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return

        # Rapport de classification
        report = classification_report(
            self.true_labels, self.predicted_labels, 
            target_names=self.classes, digits=3, zero_division=0
        )
        
        label = Label(self.metrics_inner_frame, text="Rapport de classification :", font=("Arial", 12, "bold"))
        label.pack(anchor="w", pady=(10, 0))
        
        text = tk.Text(self.metrics_inner_frame, height=15, width=80)
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)
        text.pack(pady=5)
        
        # Matrice de confusion
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                   xticklabels=self.classes, yticklabels=self.classes, ax=ax)
        ax.set_title("Matrice de confusion")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.metrics_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5)
        
        # Métriques globales
        acc = accuracy_score(self.true_labels, self.predicted_labels)
        f1 = f1_score(self.true_labels, self.predicted_labels, average='weighted')
        recall = recall_score(self.true_labels, self.predicted_labels, average='weighted')
        precision = precision_score(self.true_labels, self.predicted_labels, average='weighted')
        
        metrics_text = (
            f"Accuracy globale : {acc*100:.2f}%\n"
            f"F1-score (pondéré) : {f1:.3f}\n"
            f"Recall (rappel, pondéré) : {recall:.3f}\n"
            f"Precision (pondérée) : {precision:.3f}"
        )
        
        metrics_label = Label(self.metrics_inner_frame, text=metrics_text, font=("Arial", 12, "bold"))
        metrics_label.pack(pady=10)
        
        # Répartition des classes
        pred_counts = [self.predicted_labels.count(i) for i in range(len(self.classes))]
        true_counts = [self.true_labels.count(i) for i in range(len(self.classes))]
        
        repartition_text = "Répartition des classes (Vrai / Prédit):\n"
        for i, cls in enumerate(self.classes):
            repartition_text += f"{cls}: {true_counts[i]} / {pred_counts[i]}\n"
        
        repartition_label = Label(self.metrics_inner_frame, text=repartition_text, font=("Arial", 11))
        repartition_label.pack(pady=5)
        
        # Exemples d'erreurs
        incorrect = [i for i, (t, p) in enumerate(zip(self.true_labels, self.predicted_labels)) if t != p]
        if incorrect:
            err_label = Label(self.metrics_inner_frame, text="Exemples d'images mal classées :", font=("Arial", 11, "bold"))
            err_label.pack(pady=5)
            
            for idx in incorrect[:10]:  # Affiche les 10 premières erreurs
                img_path = self.file_paths[idx]
                true_cls = self.classes[self.true_labels[idx]]
                pred_cls = self.classes[self.predicted_labels[idx]]
                probs = self.probas[idx]
                prob_text = f"P({pred_cls})={probs[self.predicted_labels[idx]]*100:.1f}%"
                
                err_text = f"{os.path.basename(img_path)} | Vrai: {true_cls} | Prédit: {pred_cls} | {prob_text}"
                err_line = Label(self.metrics_inner_frame, text=err_text, font=("Arial", 10), fg="red")
                err_line.pack(anchor="w")

    def create_roc_curve_multiclass(self):
        """Créer les courbes ROC pour classification multiclasse"""
        for widget in self.roc_inner_frame.winfo_children():
            widget.destroy()
            
        if not self.true_labels or not self.probas:
            no_data_label = Label(self.roc_inner_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
        
        # Binariser les labels pour ROC multiclasse
        y_true = np.array(self.true_labels)
        y_scores = np.array(self.probas)
        
        # Binariser les labels vrais (one-vs-rest)
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]
        
        # Calculer ROC et AUC pour chaque classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculer micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tracer les courbes ROC pour chaque classe
        for i, (cls, color) in enumerate(zip(self.classes, self.class_colors)):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{cls} (AUC = {roc_auc[i]:.3f})')
        
        # Tracer la courbe micro-average
        ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=2,
               label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        # Ligne diagonale (classifieur aléatoire)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Classifieur aléatoire')
        
        # Formatage du graphique
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbes ROC - Classification multiclasse')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ajouter le graphique à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.roc_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistiques ROC
        stats_frame = Frame(self.roc_inner_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        auc_text = "Scores AUC par classe:\n"
        for i, cls in enumerate(self.classes):
            auc_text += f"{cls}: {roc_auc[i]:.3f}\n"
        auc_text += f"Micro-average: {roc_auc['micro']:.3f}"
        
        auc_label = Label(stats_frame, text=auc_text, font=("Arial", 12, "bold"))
        auc_label.pack(pady=5)
        
        explanation_text = (
            "Les courbes ROC montrent les performances de classification pour chaque classe.\n"
            "Le micro-average combine les performances sur toutes les classes.\n"
            "Plus l'AUC est proche de 1.0, meilleure est la performance."
        )
        
        explanation_label = Label(stats_frame, text=explanation_text, 
                                font=("Arial", 11), wraplength=600, justify=tk.LEFT)
        explanation_label.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectorApp(root)
    root.mainloop()