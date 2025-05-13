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

        # Maintenir un seuil bas pour minimiser les faux négatifs
        self.threshold = 0.02
        
        # Variables pour stocker les résultats
        self.probas = []
        self.true_labels = []
        self.predicted_labels = []
        self.file_paths = []
        self.current_image_probs = None

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

        # Ajouter un contrôle de seuil avec slider et saisie manuelle
        threshold_frame = Frame(main_frame)
        threshold_frame.pack(pady=5, fill=tk.X)

        threshold_label = Label(threshold_frame, text="Seuil de détection (plus bas = moins de faux négatifs)", font=("Arial", 11))
        threshold_label.pack(anchor="w")

        slider_frame = Frame(threshold_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        self.threshold_var = tk.DoubleVar(value=self.threshold)
        self.threshold_slider = Scale(slider_frame, from_=0.001, to=1.0,
                                    resolution=0.001, orient="horizontal",
                                    variable=self.threshold_var,
                                    length=300,
                                    command=self.threshold_changed)
        self.threshold_slider.pack(side=tk.LEFT)

        # Ajouter un champ de saisie manuelle
        self.threshold_entry = tk.Entry(slider_frame, width=8, font=("Arial", 10))
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(side=tk.LEFT, padx=5)

        # Bouton pour valider la saisie manuelle
        manual_apply_button = Button(slider_frame, text="OK", command=self.apply_manual_threshold, font=("Arial", 10))
        manual_apply_button.pack(side=tk.LEFT, padx=2)

        # Étiquette pour afficher la valeur actuelle
        threshold_value_label = Label(slider_frame, textvariable=self.threshold_var, width=5)
        threshold_value_label.pack(side=tk.LEFT, padx=5)

        apply_button = Button(slider_frame, text="Appliquer", command=self.apply_threshold, font=("Arial", 10))
        apply_button.pack(side=tk.LEFT, padx=10)
        # Information sur les faux négatifs
        fn_info = Label(threshold_frame, 
                      text="Note: Un seuil bas (ex: 0.02) minimise les faux négatifs mais peut augmenter les faux positifs.", 
                      font=("Arial", 10, "italic"), fg="gray")
        fn_info.pack(anchor="w", pady=2)

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

        # Tab pour analyse des faux négatifs
        self.fn_frame = Frame(self.notebook)
        self.notebook.add(self.fn_frame, text="Analyse des faux négatifs")

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

        # Faux négatifs frame
        self.fn_scroll_frame = Frame(self.fn_frame)
        self.fn_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fn_inner_frame = self.create_scrollable_frame(self.fn_scroll_frame)

        # Status bar
        self.status_bar = Label(self.root, text="Prêt", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def apply_manual_threshold(self):
        """Applique le seuil entré manuellement"""
        try:
            manual_value = float(self.threshold_entry.get())
            if 0 <= manual_value <= 1:
                self.threshold_var.set(manual_value)
                self.threshold = manual_value
                self.threshold_slider.set(manual_value)
                self.status_bar.config(text=f"Seuil mis à jour: {self.threshold}")
                
                # Si des données ont déjà été analysées, recalculer les prédictions
                if self.probas:
                    self.recalculate_predictions()
            else:
                self.status_bar.config(text="Erreur: Le seuil doit être entre 0 et 1")
        except ValueError:
            self.status_bar.config(text="Erreur: Valeur de seuil invalide")
            # Réinitialiser la valeur dans le champ de saisie
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold))
    def threshold_changed(self, value):
        """Callback quand le slider est manipulé"""
        # Mettre à jour le champ de saisie
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, value)

    def apply_threshold(self):
        """Applique le nouveau seuil et recalcule les prédictions"""
        try:
            new_threshold = float(self.threshold_var.get())
            if new_threshold < 0 or new_threshold > 1:
                self.status_bar.config(text="Erreur: Le seuil doit être entre 0 et 1")
                return
                
            self.threshold = new_threshold
            self.status_bar.config(text=f"Seuil mis à jour: {self.threshold}")
            # Mettre à jour le champ de saisie
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(new_threshold))
            
            # Si des données ont déjà été analysées, recalculer les prédictions
            if self.probas:
                self.recalculate_predictions()
                
        except ValueError:
            self.status_bar.config(text="Erreur: Valeur de seuil invalide")
    
    def recalculate_predictions(self):
        """Recalcule les prédictions avec le nouveau seuil"""
        self.predicted_labels = []
        for probs in self.probas:
            pred_label = int(probs[1] > self.threshold)
            self.predicted_labels.append(pred_label)
            
        # Mettre à jour les visualisations
        self.create_plot_binary()
        self.create_advanced_metrics_binary()
        self.create_roc_curve()
        self.analyze_false_negatives()
        
        # Si une image est actuellement affichée, mettre à jour son diagnostic
        if hasattr(self, 'current_image_probs') and self.current_image_probs is not None:
            pred_label = int(self.current_image_probs[1] > self.threshold)
            classes = ['NORMAL', 'MALADE']
            pred_class = classes[pred_label]
            if pred_class == 'NORMAL':
                diag = "Normal"
                color = "blue"
            else:
                diag = "Malade"
                color = "red"
            prob_text = ", ".join([f"{c}: {p*100:.2f}%" for c, p in zip(classes, self.current_image_probs)])
            self.result_label.config(text=f"Diagnostic: {diag}", fg=color)
            self.probability_label.config(text=prob_text)

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
                
                # Sauvegarder les probabilités pour l'image actuelle
                self.current_image_probs = probs
                
                classes = ['NORMAL', 'MALADE']
                pred_label = int(probs[1] > self.threshold)  # Utilise le seuil actuel
                pred_class = classes[pred_label]
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
                        pred_label = int(probs[1] > self.threshold)  # Utilise le seuil actuel
                        self.true_labels.append(true_label)
                        self.predicted_labels.append(pred_label)
                        self.probas.append(probs)
                        self.file_paths.append(file_path)
            self.create_plot_binary()
            self.create_advanced_metrics_binary()
            self.create_roc_curve()
            self.analyze_false_negatives()
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
        
        # Ajouter une ligne verticale pour le seuil actuel
        ax.axvline(self.threshold, color='red', linestyle='-', linewidth=2)
        ax.text(self.threshold + 0.02, ax.get_ylim()[1]*0.8, f"Seuil: {self.threshold:.3f}", 
                color='red', fontweight='bold')
        
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
        
        # Calcul des faux négatifs et faux positifs
        tn, fp, fn, tp = cm.ravel()
        
        # Métriques globales
        acc = accuracy_score(self.true_labels, self.predicted_labels)
        f1 = f1_score(self.true_labels, self.predicted_labels, average='weighted')
        recall = recall_score(self.true_labels, self.predicted_labels, average='weighted')
        precision = precision_score(self.true_labels, self.predicted_labels, average='weighted')
        
        # Métriques spécifiques aux faux négatifs
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics_text = (
            f"Accuracy globale : {acc*100:.2f}%\n"
            f"F1-score (pondéré) : {f1:.3f}\n"
            f"Recall (rappel, pondéré) : {recall:.3f}\n"
            f"Precision (pondérée) : {precision:.3f}\n\n"
            f"Taux de faux négatifs (FNR) : {false_negative_rate:.3f}\n"
            f"Taux de faux positifs (FPR) : {false_positive_rate:.3f}\n"
            f"Nombre de faux négatifs : {fn} / {fn + tp} cas malades\n"
            f"Nombre de faux positifs : {fp} / {fp + tn} cas normaux"
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
    def analyze_false_negatives(self):
        """Analyse les cas de faux négatifs pour aider à comprendre les erreurs critiques"""
        for widget in self.fn_inner_frame.winfo_children():
            widget.destroy()
            
        if not self.true_labels or not self.probas:
            no_data_label = Label(self.fn_inner_frame, text="Aucune donnée à afficher", font=("Arial", 14))
            no_data_label.pack(pady=20)
            return
            
        # Identifier les faux négatifs (vrai=1, prédit=0)
        false_negatives = [i for i, (true, pred) in enumerate(zip(self.true_labels, self.predicted_labels)) 
                            if true == 1 and pred == 0]
        
        # Section d'information
        info_label = Label(self.fn_inner_frame, 
                        text=f"Analyse des faux négatifs (cas malades non détectés): {len(false_negatives)} trouvés",
                        font=("Arial", 12, "bold"))
        info_label.pack(pady=10)
        
        if not false_negatives:
            no_fn_label = Label(self.fn_inner_frame, text="Aucun faux négatif trouvé - excellent!", 
                            font=("Arial", 12), fg="green")
            no_fn_label.pack(pady=10)
            return
        
        # Afficher les statistiques de probabilités pour les faux négatifs
        probs_fn = [self.probas[i][1] for i in false_negatives]  # Probabilités pour la classe positive
        
        stats_text = (
            f"Statistiques des probabilités pour les faux négatifs:\n"
            f"Probabilité moyenne: {np.mean(probs_fn):.3f}\n"
            f"Probabilité min: {np.min(probs_fn):.3f}, max: {np.max(probs_fn):.3f}\n"
            f"Distance moyenne au seuil: {self.threshold - np.mean(probs_fn):.3f}"
        )
        
        stats_label = Label(self.fn_inner_frame, text=stats_text, font=("Arial", 11))
        stats_label.pack(pady=5)
        
        # Afficher la distribution des probabilités des faux négatifs
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(probs_fn, bins=20, color="red", kde=True, ax=ax)
        ax.axvline(self.threshold, color='black', linestyle='--', linewidth=2, 
                    label=f'Seuil actuel: {self.threshold:.3f}')
        ax.set_title('Distribution des probabilités pour les faux négatifs')
        ax.set_xlabel('Probabilité de la classe MALADE')
        ax.set_ylabel('Fréquence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.fn_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        
        # Afficher quelques exemples de faux négatifs
        examples_label = Label(self.fn_inner_frame, text="Exemples de faux négatifs:", font=("Arial", 11, "bold"))
        examples_label.pack(pady=5, anchor="w")
        
        # Limiter à 10 exemples maximum
        for idx in false_negatives[:10]:
            img_path = self.file_paths[idx]
            prob = self.probas[idx][1] * 100  # Probabilité pour la classe positive en %
            example_text = f"{os.path.basename(img_path)} | Probabilité MALADE: {prob:.2f}% | Seuil: {self.threshold*100:.2f}%"
            example_label = Label(self.fn_inner_frame, text=example_text, font=("Arial", 10))
            example_label.pack(anchor="w")
        
        # Recommandations
        recom_frame = Frame(self.fn_inner_frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        recom_frame.pack(fill=tk.X, pady=10)
        
        recom_title = Label(recom_frame, text="Recommandations pour réduire les faux négatifs:", 
                        font=("Arial", 11, "bold"))
        recom_title.pack(anchor="w", pady=5)
        
        recommendations = [
            "Diminuer le seuil de détection pour augmenter la sensibilité",
            "Ajouter plus d'exemples de pneumonie dans l'ensemble d'entraînement",
            "Utiliser l'augmentation de données sur les cas de pneumonie",
            "Envisager un modèle avec une meilleure sensibilité (même au prix de faux positifs)"
        ]
        
        for rec in recommendations:
            rec_label = Label(recom_frame, text="• " + rec, font=("Arial", 10), wraplength=600, justify=tk.LEFT)
            rec_label.pack(anchor="w", pady=2)

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
        
        # Marquer le seuil actuel sur la courbe ROC
        # Trouver l'index du seuil le plus proche de notre seuil actuel
        threshold_diff = np.abs(thresholds - self.threshold)
        closest_idx = np.argmin(threshold_diff)
        
        if closest_idx < len(fpr) and closest_idx < len(tpr):
            ax.plot(fpr[closest_idx], tpr[closest_idx], 'ro', markersize=8, 
                    label=f'Seuil actuel = {self.threshold:.3f}')
        
        # Find index for high sensitivity (minimize false negatives)
        # Nous voulons trouver un bon compromis entre sensibilité élevée (>0.95) et FPR raisonnable
        target_sensitivity = 0.95
        # Trouver les indices où TPR (sensibilité) >= 0.95
        high_sensitivity_indices = np.where(tpr >= target_sensitivity)[0]
        
        # Définir best_threshold et best_idx (correction)
        best_idx = closest_idx  # Valeur par défaut
        best_threshold = self.threshold  # Valeur par défaut
        
        if len(high_sensitivity_indices) > 0:
            # Parmi ces indices, trouver celui qui minimise FPR
            optimal_idx = high_sensitivity_indices[np.argmin(fpr[high_sensitivity_indices])]
            optimal_threshold = thresholds[optimal_idx]
            
            # Mettre à jour best_idx et best_threshold
            best_idx = optimal_idx
            best_threshold = optimal_threshold
            
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=8, 
                    label=f'Seuil recommandé = {optimal_threshold:.3f}')
        
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