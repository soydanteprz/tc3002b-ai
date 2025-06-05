#!/usr/bin/env python
"""
Plagiarism detection using Machine Learning
TF-IDF (n-gramas) and Logistic Regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')
from astcc import *




class MLPlagiarismClassifier:
    """
    Plagiarism detection classifier using Machine Learning
    """

    def __init__(self, ngram_range=(1, 2), max_features=5000):
        """
        Initialice the classifier with TF-IDF and Logistic Regression
        :param ngram_range: Range of n-grams to use (default: unigrams and bigrams)
        :param max_features: Maximum number of features to extract
        :return: None
        """
        self.ngram_range = ngram_range
        self.max_features = max_features

        # Use TFIDFPlagiarismDetector instead of direct TfidfVectorizer
        self.tfidf_detector = TFIDFPlagiarismDetector()

        # Configure the vectorizer inside the detector
        self.tfidf_detector.vectorizer.ngram_range = ngram_range
        self.tfidf_detector.vectorizer.max_features = max_features

        # Keep reference to the vectorizer for convenience
        self.vectorizer = self.tfidf_detector.vectorizer

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False

    def load_dataset(self, base_path: str, split: str) -> Tuple[List[str], List[int], pd.DataFrame]:
        """
        Loads the dataset from CSV files and prepares the text and labels
        :param base_path: Base path where the dataset is located
        :param split: Split of the dataset ('train', 'validation', 'test')
        :return: Tuple with texts, labels and DataFrame with valid indices
        """
        csv_path = os.path.join(base_path, f'{split}.csv')
        df = pd.read_csv(csv_path)

        if 'source_dataset' in df.columns:
            df = df[df['source_dataset'].isin(['ir_plag', 'conplag'])]

        texts = []
        labels = []
        valid_indices = []

        print(f"\nCargando datos de {split}...")

        for idx, row in df.iterrows():
            dataset = row.get('source_dataset', 'unknown')
            label = row['label']

            if dataset == 'ir_plag':
                folder = row.get('folder_name')
                if folder:
                    folder_path = os.path.join(base_path, split, folder)
                    path1 = os.path.join(folder_path, 'original.java')
                    path2 = os.path.join(folder_path, 'compared.java')
                else:
                    continue

            elif dataset == 'conplag':
                file1 = row.get('file1')
                file2 = row.get('file2')

                if file1 and file2:
                    file1_base = os.path.splitext(file1)[0]
                    file2_base = os.path.splitext(file2)[0]

                    folder1 = os.path.join(base_path, split, f"{file1_base}_{file2_base}")
                    folder2 = os.path.join(base_path, split, f"{file2_base}_{file1_base}")

                    if os.path.isdir(folder1):
                        folder_path = folder1
                    elif os.path.isdir(folder2):
                        folder_path = folder2
                    else:
                        continue

                    path1 = os.path.join(folder_path, file1)
                    path2 = os.path.join(folder_path, file2)
                else:
                    continue
            else:
                continue

            if os.path.exists(path1) and os.path.exists(path2):
                try:
                    with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
                        text1 = f.read()
                    with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
                        text2 = f.read()

                    combined_text = text1 + " [SEP] " + text2

                    texts.append(combined_text)
                    labels.append(label)
                    valid_indices.append(idx)

                except Exception as e:
                    print(f"Error leyendo archivos: {e}")
                    continue

        print(f"Cargados {len(texts)} pares de archivos")
        print(f"Plagios: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"No plagios: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")

        df_valid = df.iloc[valid_indices].copy()
        df_valid['combined_text'] = texts

        return texts, labels, df_valid

    def train(self, X_train: List[str], y_train: List[int],
              X_val: List[str] = None, y_val: List[int] = None,
              optimize_hyperparameters: bool = False):
        """
        Trains the model with the provided training data
        :param X_train: List of training texts
        :param y_train: List of training labels
        :param X_val: List of validation texts
        :param y_val: List of validation labels
        :param optimize_hyperparameters: Whether to optimize hyperparameters using grid search
        :return: None
        """
        print("\n Entrenando el modelo")

        # Vectorize the training data
        print("Vectorizando con TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Dimensiones: {X_train_tfidf.shape}")
        print(f"Características extraídas: {X_train_tfidf.shape[1]}")

        if optimize_hyperparameters and X_val is not None:
            print("Optimizando hiperparámetros")

            # Grid search for hyperparameter tuning
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

            grid_search = GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=42),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train_tfidf, y_train)
            self.model = grid_search.best_estimator_

            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score CV: {grid_search.best_score_:.3f}")

        else:
            self.model.fit(X_train_tfidf, y_train)

        self.is_trained = True
        print("Modelo entrenado")

        if X_val is not None and y_val is not None:
            X_val_tfidf = self.vectorizer.transform(X_val)
            y_val_pred = self.model.predict(X_val_tfidf)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"\nAccuracy en validación: {val_accuracy:.3f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Makes predictions on the provided texts
        :param texts: List of texts to predict
        :return: Array of predictions
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Makes probability predictions on the provided texts
        :param texts: List of texts to predict probabilities
        :return: Array of probabilities
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)

    def evaluate(self, X_test: List[str], y_test: List[int],
                 dataset_name: str = "Test") -> Dict:
        """
        Evaluates the model on the provided test data
        :param X_test: List of test texts
        :param y_test: List of test labels
        :param dataset_name: Name of the dataset for reporting
        :return: Dictionary with evaluation metrics
        """
        print(f"\nEvaluando en {dataset_name}...")

        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        y_proba = self.model.predict_proba(X_test_tfidf)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{dataset_name} Classification Report:")
        print("=" * 60)
        print(classification_report(y_test, y_pred,
                                    target_names=['No Plagio', 'Plagio'],
                                    digits=3))

        # Calculate AUC-ROC score
        try:
            auc_score = roc_auc_score(y_test, y_proba)
            print(f"AUC-ROC Score: {auc_score:.3f}")
        except:
            auc_score = None

        self._plot_confusion_matrix(y_test, y_pred, dataset_name)

        # Roc
        if auc_score is not None:
            self._plot_roc_curve(y_test, y_proba, dataset_name)

        results = pd.DataFrame({
            'TrueLabel': y_test,
            'Predicted': y_pred,
            'Probability': y_proba
        })

        output_file = f"csv/ml_results_{dataset_name.lower()}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n Resultados guardados en: {output_file}")

        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def _plot_confusion_matrix(self, y_true, y_pred, dataset_name):
        """
        Generates and saves confusion matrix plot
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param dataset_name: Name of the dataset for the plot title
        :return: None
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Plagio', 'Plagio'],
                    yticklabels=['No Plagio', 'Plagio'])
        plt.title(f"Matriz de Confusión - Logistic Regression ({dataset_name})")
        plt.xlabel("Predicho")
        plt.ylabel("Real")

        filename = f"images/ml_confusion_matrix_{dataset_name.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Matriz de confusión guardada en: {filename}")

    def _plot_roc_curve(self, y_true, y_proba, dataset_name):
        """Genera y guarda curva ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Logistic Regression ({dataset_name})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filename = f"images/ml_roc_curve_{dataset_name.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Curva ROC guardada en: {filename}")

    def save_model(self, filepath: str = "ml_plagiarism_model.pkl"):
        """
        Saves the trained model to a file
        :param filepath: Path to save the model
        :return: None
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features
        }

        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str = "ml_plagiarism_model.pkl"):
        """
        Loads a trained model from a file
        :param filepath: Path to the model file
        :return: None
        """
        model_data = joblib.load(filepath)

        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.ngram_range = model_data['ngram_range']
        self.max_features = model_data['max_features']
        self.is_trained = True

        print(f"Modelo cargado desde: {filepath}")

    def analyze_feature_importance(self, top_n: int = 20):
        """
        Analyzes and visualizes the most important features for the model
        :param top_n: Number of top features to display
        :return: None
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

        plt.figure(figsize=(10, 8))

        top_positive = feature_importance[feature_importance['coefficient'] > 0].head(top_n)
        top_negative = feature_importance[feature_importance['coefficient'] < 0].head(top_n)

        top_features = pd.concat([top_positive, top_negative]).sort_values('coefficient')

        colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coeficiente')
        plt.title(f'Top {top_n} Caracteristicas mas importantes')
        plt.tight_layout()

        plt.savefig('images/ml_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nTop {top_n} caracteristicas para detectar plagio:")
        for _, row in top_positive.head(10).iterrows():
            print(f"{row['feature']}: {row['coefficient']:.3f}")

        print(f"\nTop {top_n} caracteristicas para detectar NO plagio:")
        for _, row in top_negative.head(10).iterrows():
            print(f"{row['feature']}: {row['coefficient']:.3f}")


def main():
    """
    Main function to run the plagiarism detection system
    """
    print("SISTEMA DE CLASIFICACION DE PLAGIO")
    print("=" * 70)
    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("csv", exist_ok=True)

    BASE_PATH = "data/splits"

    classifier = MLPlagiarismClassifier(
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        max_features=5000
    )

    print("\nCARGANDO DATASETS")
    X_train, y_train, df_train = classifier.load_dataset(BASE_PATH, 'train')
    X_val, y_val, df_val = classifier.load_dataset(BASE_PATH, 'validation')
    X_test, y_test, df_test = classifier.load_dataset(BASE_PATH, 'test')

    print("\nENTRENAMIENTO")
    classifier.train(
        X_train, y_train,
        X_val, y_val,
        optimize_hyperparameters=True  # Optimize hyperparameters
    )

    print("\nEVALUACIÓN EN VALIDACIÓN")
    val_results = classifier.evaluate(X_val, y_val, "Validation")

    print("\nEVALUACIÓN EN TEST")
    test_results = classifier.evaluate(X_test, y_test, "Test")

    print("\nANÁLISIS DE CARACTERÍSTICAS")
    classifier.analyze_feature_importance(top_n=20)

    classifier.save_model("models/ml_plagiarism_model.pkl")

    print("\nRESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"Validation Accuracy: {val_results['accuracy']:.3f}")
    print(f"Test Accuracy: {test_results['accuracy']:.3f}")
    if test_results['auc']:
        print(f"Test AUC-ROC: {test_results['auc']:.3f}")

    summary = pd.DataFrame({
        'Método': ['Logistic Regression (TF-IDF n-grams)'],
        'Val_Accuracy': [val_results['accuracy']],
        'Test_Accuracy': [test_results['accuracy']],
        'Test_AUC': [test_results['auc'] if test_results['auc'] else 'N/A']
    })

    summary.to_csv('csv/ml_summary_results.csv', index=False)
    print("\nResumen guardado en: ml_summary_results.csv")

    print("\nProceso completado")


if __name__ == "__main__":
    main()
