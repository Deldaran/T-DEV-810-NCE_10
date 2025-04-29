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
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch.nn as nn

matplotlib.use("TkAgg")

class ResNet18WithDropout(nn.Module):
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
        self.root.title("Détecteur de Pneumonie")
        self.root.geometry("1100x800")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

        self.probas = []
        self.true_labels = []
        self.predicted_labels = []
        self.file_paths = []

        self.create_widgets()

    def load_model(self):
        try:
            checkpoint = torch.load('models/final_model.pth', map_location=self.device, weights_only=False)
            
            # Print checkpoint structure for diagnosis
            print("Checkpoint keys:", checkpoint.keys())
            
            # Get the state dict from the checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Print first few keys to understand model structure
            print("First 5 state dict keys:", list(state_dict.keys())[:5])
            
            # Continue with ResNet18 but note it will likely fail
            base_model = models.resnet18(weights=None)
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = ResNet18WithDropout(base_model, num_classes=2, dropout_p=0.5)
            
            try:
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Failed to load state dict: {e}")
                # The model architecture doesn't match - will need to get the correct architecture
                self.model = None
                return
                
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Modèle chargé avec succès")
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

        title_label = Label(main_frame, text="Détecteur de Pneumonie par Rayons X", font=("Arial", 18, "bold"))
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

        # New tab for ROC curve
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
            display_image = Image.open(file_path)
            display_image = display_image.resize((300, 300), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

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
                classes = ['NORMAL', 'MALADE']
                pred_idx = probs.argmax()
                pred_class = classes[pred_idx]
                if pred_class == 'NORMAL':
                    diag = "Normal"
                    color = "blue"
                else:
                    diag = "Malade"
                    color = "red"
                prob_text = ", ".join([f"{c}: {p*100:.2f}%" for c, p in zip(classes, probs)])
                self.result_label.config(text=f"Diagnostic: {diag}", fg=color)
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
                        fname = file.lower()
                        parent_folder = os.path.basename(root_dir).upper()
                        if parent_folder == 'NORMAL' or "normal" in fname:
                            true_label = 0
                        else:  # bacteria ou virus => malade
                            true_label = 1
                        probs = self.process_single_image_for_batch(file_path)
                        pred_label = probs.argmax()
                        self.true_labels.append(true_label)
                        self.predicted_labels.append(pred_label)
                        self.probas.append(probs)
                        self.file_paths.append(file_path)
            self.create_plot_binary()
            self.create_advanced_metrics_binary()
            self.create_roc_curve()  # Add ROC curve creation
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
            return np.array([0.0, 0.0])

    def create_plot_binary(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        if not self.probas:
            no_data_label = Label(self.graph_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
        probas = np.array(self.probas)
        classes = ['NORMAL', 'MALADE']
        colors = sns.color_palette("Set2", n_colors=2)
        fig, ax = plt.subplots(figsize=(10, 7))
        bins = np.linspace(0, 1, 25)
        for i, (cls, color) in enumerate(zip(classes, colors)):
            sns.histplot(probas[:, i], bins=bins, color=color, label=cls, kde=False, ax=ax, alpha=0.6)
            mean_val = np.mean(probas[:, i])
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=1)
            ax.text(mean_val + 0.02, ax.get_ylim()[1]*0.9, f"Mean {cls}: {mean_val:.2f}", color=color)
        ax.legend(title="Classes")
        ax.set_title('Distribution des probabilités par classe')
        ax.set_xlabel('Probabilité')
        ax.set_ylabel("Nombre d'images")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        stats_frame = Frame(self.graph_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        means = [np.mean(probas[:, i]) for i in range(2)]
        stats_text = f"Moyenne NORMAL: {means[0]:.2f} | MALADE: {means[1]:.2f}"
        stats_label = Label(stats_frame, text=stats_text, font=("Arial", 12))
        stats_label.pack()

    def create_advanced_metrics_binary(self):
        for widget in self.metrics_inner_frame.winfo_children():
            widget.destroy()
        if not self.true_labels or not self.predicted_labels:
            no_data_label = Label(self.metrics_inner_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
        classes = ['NORMAL', 'MALADE']
        # Rapport de classification
        report = classification_report(
            self.true_labels, self.predicted_labels, target_names=classes, digits=3, zero_division=0
        )
        label = Label(self.metrics_inner_frame, text="Rapport de classification :", font=("Arial", 12, "bold"))
        label.pack(anchor="w", pady=(10, 0))
        text = tk.Text(self.metrics_inner_frame, height=10, width=80)
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)
        text.pack(pady=5)
        # Matrice de confusion avec seaborn
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
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
        # Répartition des classes (vrai / prédit)
        pred_counts = [self.predicted_labels.count(i) for i in range(len(classes))]
        true_counts = [self.true_labels.count(i) for i in range(len(classes))]
        repartition_text = "Répartition des classes (Vrai / Prédit):\n"
        for i, cls in enumerate(classes):
            repartition_text += f"{cls}: {true_counts[i]} / {pred_counts[i]}\n"
        repartition_label = Label(self.metrics_inner_frame, text=repartition_text, font=("Arial", 11))
        repartition_label.pack(pady=5)
        # Affichage des erreurs (images mal classées)
        incorrect = [i for i, (t, p) in enumerate(zip(self.true_labels, self.predicted_labels)) if t != p]
        if incorrect:
            err_label = Label(self.metrics_inner_frame, text="Exemples d'images mal classées :", font=("Arial", 11, "bold"))
            err_label.pack(pady=5)
            for idx in incorrect[:5]:  # Affiche les 5 premières erreurs
                img_path = self.file_paths[idx]
                true_cls = classes[self.true_labels[idx]]
                pred_cls = classes[self.predicted_labels[idx]]
                err_text = f"{os.path.basename(img_path)} | Vrai: {true_cls} | Prédit: {pred_cls}"
                err_line = Label(self.metrics_inner_frame, text=err_text, font=("Arial", 10), fg="red")
                err_line.pack(anchor="w")

    def create_roc_curve(self):
        """Create ROC curve visualization"""
        for widget in self.roc_inner_frame.winfo_children():
            widget.destroy()
            
        if not self.true_labels or not self.probas:
            no_data_label = Label(self.roc_inner_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
            
        # Convert probabilities to numpy array
        y_true = np.array(self.true_labels)
        y_scores = np.array(self.probas)[:, 1]  # Use probability for the positive class (MALADE)
        
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        # Add annotations for selected thresholds
        # Only show a subset of thresholds to avoid cluttering
        threshold_indices = np.linspace(0, len(thresholds) - 1, 5, dtype=int)
        for i in threshold_indices:
            if i < len(thresholds) and i < len(fpr) and i < len(tpr):
                ax.annotate(f"{thresholds[i]:.2f}", 
                           xy=(fpr[i], tpr[i]), 
                           xytext=(fpr[i]+0.05, tpr[i]-0.05),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        # Add optimization point (best threshold)
        # Find best threshold based on Youden's J statistic (tpr - fpr)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        ax.plot(fpr[best_idx], tpr[best_idx], 'ro', markersize=8, 
                label=f'Optimal threshold = {best_threshold:.3f}')
        
        # Plot formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de faux positifs (1 - Spécificité)')
        ax.set_ylabel('Taux de vrais positifs (Sensibilité)')
        ax.set_title('Courbe ROC (Receiver Operating Characteristic)')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add the plot to the frame
        canvas = FigureCanvasTkAgg(fig, master=self.roc_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add additional ROC statistics
        stats_frame = Frame(self.roc_inner_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculate additional metrics at the optimal threshold
        y_pred_optimal = (y_scores >= best_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
        
        # Display metrics
        roc_stats_text = (
            f"AUC (Area Under Curve): {roc_auc:.3f}\n"
            f"Seuil optimal: {best_threshold:.3f}\n"
            f"Accuracy au seuil optimal: {optimal_accuracy:.3f}\n"
            f"Sensibilité (TPR) au seuil optimal: {tpr[best_idx]:.3f}\n"
            f"Spécificité au seuil optimal: {1-fpr[best_idx]:.3f}"
        )
        
        roc_explanation = (
            "La courbe ROC montre le compromis entre la sensibilité (taux de vrais positifs) "
            "et la spécificité (1 - taux de faux positifs) pour différents seuils de classification.\n"
            "Un modèle parfait aurait une AUC de 1.0, tandis qu'un modèle aléatoire aurait une AUC de 0.5."
        )
        
        stats_label = Label(stats_frame, text=roc_stats_text, font=("Arial", 12, "bold"))
        stats_label.pack(pady=5)
        
        explanation_label = Label(stats_frame, text=roc_explanation, font=("Arial", 11), wraplength=600, justify=tk.LEFT)
        explanation_label.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectorApp(root)
    root.mainloop()