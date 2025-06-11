import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

class PlagiarismVisualizationTools:
    """
    Provides tools for visualizing model evaluation results, including confusion matrices,
    ROC curves, Precision-Recall curves, metrics comparison, and feature importance.

    The class allows for a detailed visual comparison of multiple machine learning models'
    performance metrics using various types of plots. It supports operations such as rendering
    and saving visualization outputs for confusion matrices, ROC/Precision-Recall curves,
    bar charts of model metrics, and feature importance.

    Usage of this class improves the interpretability of models and aids in decision-making
    based on their comparative performance measures.

    :ivar figsize: Default figure size for plots.
    :ivar colors: Default color palette used for plots.
    """

    def __init__(self, figsize=(12, 8)):
        """
        This class provides a blueprint for a customizable plotting setup. It initializes
        the default figure size for plots, sets up a predefined color palette for graph
        elements, and applies specific styles using matplotlib and seaborn libraries. The
        styles and configurations are tailored for a visually cohesive and standardized
        output for data visualization.

        :param figsize: A tuple that specifies the dimensions of the figure in inches,
            default is (12, 8).
        :type figsize: tuple
        """
        self.figsize = figsize
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
        plt.style.use('default')
        sns.set_palette("husl")

    def compare_confusion_matrices(self, results_dict, dataset_name="Test"):
        """
        Compares confusion matrices for multiple classification models and visualizes them
        side-by-side using heatmaps. The method computes essential classification metrics
        like accuracy, precision, recall, and F1 score for each model and includes them
        in the heatmap titles. Each heatmap represents the true labels and predicted labels
        of a given model's results.

        The function generates a subplot for each model with a confusion matrix heatmap.
        It also saves the resulting visualization as a PNG file in the `images` directory.

        :param results_dict: A dictionary where keys are model names and values are
            dictionaries containing `y_true` (true class labels) and `y_pred` (predicted class labels).
        :param dataset_name: A string indicating the name of the dataset being analyzed.
            Defaults to "Test".
        :return: None
        """
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred = results['y_pred']

            cm = confusion_matrix(y_true, y_pred)

            # Calcular métricas
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Crear heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['No Plagio', 'Plagio'],
                        yticklabels=['No Plagio', 'Plagio'])

            axes[i].set_title(f'{model_name}\nAcc: {accuracy:.3f} | F1: {f1:.3f}',
                              fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')

        plt.suptitle(f'Comparación de Matrices de Confusión - {dataset_name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'images/improvement/comparison_confusion_matrices_{dataset_name.lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def compare_roc_curves(self, results_dict, dataset_name="Test"):
        """
        Plots and compares the ROC curves for multiple models and saves the
        resulting image to a file while also displaying it. Each model's ROC
        curve is plotted with its respective AUC (Area Under the Curve) value.

        :param results_dict: Dictionary containing model results where each
            key is the model name, and its value is a dictionary with keys
            `y_true` (list or array of true labels) and `y_proba` (list or
            array of predicted probabilities for the positive class).
        :type results_dict: dict
        :param dataset_name: Name of the dataset used for analysis, defaulting
            to "Test". The name is incorporated into the plot title and the
            saved file name.
        :type dataset_name: str
        :return: None
        """
        plt.figure(figsize=self.figsize)

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_proba = results['y_proba']

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)],
                     lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
        plt.title(f'Comparación de Curvas ROC - {dataset_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'images/improvement/comparison_roc_curves_{dataset_name.lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def compare_precision_recall_curves(self, results_dict, dataset_name="Test"):
        """
        Generates and compares precision-recall curves for different models provided in the
        results dictionary. It calculates and plots the precision and recall metrics along
        with the average precision for each model, enabling a visual assessment of model
        performance.

        :param results_dict: Dictionary where keys are model names and values are dictionaries
            containing 'y_true' and 'y_proba', representing true labels and predicted
            probabilities, respectively.
        :type results_dict: dict
        :param dataset_name: Name of the dataset to include in plot titles and saved file names.
            Defaults to "Test".
        :type dataset_name: str, optional
        :return: None
        """
        plt.figure(figsize=self.figsize)

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_proba = results['y_proba']

            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)

            plt.plot(recall, precision, color=self.colors[i % len(self.colors)],
                     lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Comparación de Curvas Precision-Recall - {dataset_name}',
                  fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'images/improvement/comparison_pr_curves_{dataset_name.lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def compare_metrics_bar_chart(self, metrics_dict):
        """
        Generates and displays a bar chart to compare various metrics for different models.

        This function takes a dictionary of metrics, where the keys represent the model names
        and the values are dictionaries of metric scores associated with each model. It ensures
        that all required metrics ('accuracy', 'precision', 'recall', 'f1', 'auc') are present
        for comparison. Missing metrics are set to a default value of 0.0. The bar chart is
        structured in a 2x3 grid layout, where each subplot displays one of the required metrics.
        Additionally, the sixth subplot is unused and hidden. The function saves the resulting
        bar chart as an image file and displays it.

        :param metrics_dict: A dictionary where the keys are model names (str) and the values
            are dictionaries that map metric names (e.g., 'accuracy', 'precision')
            to their corresponding float values. Missing values will default to 0.0.
        :return: None
        """

        df = pd.DataFrame(metrics_dict).T

        # Asegurar que tenemos las métricas básicas
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in required_metrics:
            if metric not in df.columns:
                df[metric] = 0.0

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            if metric in df.columns:
                bars = axes[i].bar(df.index, df[metric], color=self.colors[i % len(self.colors)], alpha=0.8)
                axes[i].set_title(f'{label}', fontsize=14, fontweight='bold')
                axes[i].set_ylim(0, 1.1)
                axes[i].grid(True, alpha=0.3)

                # Añadir valores en las barras
                for bar, value in zip(bars, df[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

                # Rotar etiquetas si son largas
                if any(len(str(x)) > 8 for x in df.index):
                    axes[i].tick_params(axis='x', rotation=45)

        # Ocultar el subplot extra
        axes[5].axis('off')

        plt.suptitle('Comparación de Métricas por Modelo', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/improvement/comparison_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance_comparison(self, importance_data, top_n=20):
        """
        Plots the comparison of feature importance for two types of features - "ASTCC" and "TF-IDF"
        - displaying the top N features for each type. The plot is saved as an image and also displayed
        by the method.

        The `importance_data` parameter is expected to contain the columns `feature`,
        `abs_importance`, and `coefficient`. The comparison separately highlights relevant features
        belonging to `astcc_` and `tfidf_` prefixes, and the top features are illustrated through
        bar charts.

        :param importance_data: A DataFrame containing information about feature importance.
                                Should include `feature` as a column indicating feature names,
                                `abs_importance` representing their absolute importance, and
                                `coefficient` indicating their associated coefficients for color coding.
        :type importance_data: pandas.DataFrame
        :param top_n: The maximum number of features to display collectively for both "ASTCC"
                      and "TF-IDF" types. Defaults to 20 and splits equally between the two types.
        :type top_n: int, optional
        :return: None, as the function does not return any value but directly saves and
                 displays the plotted feature importance comparison.
        :rtype: None
        """


        astcc_features = importance_data[importance_data['feature'].str.startswith('astcc_')].head(top_n//2)
        tfidf_features = importance_data[importance_data['feature'].str.startswith('tfidf_')].head(top_n//2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Características AST
        if not astcc_features.empty:
            y_pos = np.arange(len(astcc_features))
            bars1 = ax1.barh(y_pos, astcc_features['abs_importance'],
                             color=['green' if x > 0 else 'red' for x in astcc_features['coefficient']],
                             alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([f.replace('astcc_', '') for f in astcc_features['feature']], fontsize=9)
            ax1.set_xlabel('Importancia Absoluta')
            ax1.set_title(f'Top {len(astcc_features)} Características AST', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # Características TF-IDF
        if not tfidf_features.empty:
            y_pos = np.arange(len(tfidf_features))
            bars2 = ax2.barh(y_pos, tfidf_features['abs_importance'],
                             color=['blue' if x > 0 else 'orange' for x in tfidf_features['coefficient']],
                             alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f.replace('tfidf_', '') for f in tfidf_features['feature']], fontsize=9)
            ax2.set_xlabel('Importancia Absoluta')
            ax2.set_title(f'Top {len(tfidf_features)} Características TF-IDF', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Comparación de Importancia: AST vs TF-IDF', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/improvement/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_probability_distributions(self, results_dict, dataset_name="Test"):
        """
        Generates and visualizes histograms of probability distributions for each class
        (No Plagiarism and Plagiarism) based on predictions from multiple models. Each
        model's results are displayed as a pair of histograms, and the visualization
        provides insight into the distribution of probabilities for true and false
        predictions across different models. The distribution plots are saved as an
        image file using the dataset name for reference.

        :param results_dict: A dictionary where the keys are model names (str) and
            the values are dictionaries containing the true labels ('y_true') as a
            numpy array or list and the predicted probabilities ('y_proba') as a numpy
            array or list.
        :param dataset_name: Name of the dataset (default: "Test") used for the title
            of the visualization and the output image filename.
        :type dataset_name: str
        :return: None. Generates a plot and saves the image while displaying it.
        """

        n_models = len(results_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_proba = results['y_proba']

            # Probabilidades para cada clase
            proba_no_plagio = y_proba[np.array(y_true) == 0]
            proba_plagio = y_proba[np.array(y_true) == 1]

            # Histograma para No Plagio
            axes[0, i].hist(proba_no_plagio, bins=30, alpha=0.7, color='blue',
                            label=f'No Plagio (n={len(proba_no_plagio)})')
            axes[0, i].set_title(f'{model_name} - No Plagio')
            axes[0, i].set_xlabel('Probabilidad de Plagio')
            axes[0, i].set_ylabel('Frecuencia')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()

            # Histograma para Plagio
            axes[1, i].hist(proba_plagio, bins=30, alpha=0.7, color='red',
                            label=f'Plagio (n={len(proba_plagio)})')
            axes[1, i].set_title(f'{model_name} - Plagio')
            axes[1, i].set_xlabel('Probabilidad de Plagio')
            axes[1, i].set_ylabel('Frecuencia')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()

        plt.suptitle(f'Distribución de Probabilidades por Clase - {dataset_name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'images/improvement/probability_distributions_{dataset_name.lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_comparison_report(self, results_dict, feature_importance=None,
                                               dataset_name="Test", save_report=True):
        """
        Generates a comprehensive comparison report for model evaluation using various metrics
        and visualizations, including confusion matrices, ROC curves, precision-recall curves,
        probability distributions, metrics bar chart, and optionally feature importance analysis.
        A summary of numerical results is also provided, with an option to save the report.

        :param results_dict: Dictionary containing model results. Each key is a model's name,
            and the value is a dictionary containing 'y_true', 'y_pred', and 'y_proba'.
        :param feature_importance: (Optional) Data for feature importance analysis for the models.
        :param dataset_name: Name of the dataset for which the report is being generated.
            Defaults to "Test".
        :param save_report: Whether to save the summary as a CSV file. Defaults to True.
        :return: A pandas DataFrame with the computed metrics for all models.
        """

        print(f"Generando reporte completo de comparación - {dataset_name}")
        print("="*60)

        # 1. Matrices de confusión
        print("Generando matrices de confusión...")
        self.compare_confusion_matrices(results_dict, dataset_name)

        # 2. Curvas ROC
        print("Generando curvas ROC...")
        self.compare_roc_curves(results_dict, dataset_name)

        # 3. Curvas Precision-Recall
        print("Generando curvas Precision-Recall...")
        self.compare_precision_recall_curves(results_dict, dataset_name)

        # 4. Distribuciones de probabilidad
        print("Generando distribuciones de probabilidad...")
        self.plot_probability_distributions(results_dict, dataset_name)

        # 5. Gráfico de barras de métricas
        print("Generando comparación de métricas...")
        metrics_dict = {}
        for model_name, results in results_dict.items():
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_proba = results['y_proba']

            # Calcular métricas
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            metrics_dict[model_name] = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'auc': auc(*roc_curve(y_true, y_proba)[:2])
            }

        self.compare_metrics_bar_chart(metrics_dict)

        # 6. Importancia de características (si se proporciona)
        if feature_importance is not None:
            self.plot_feature_importance_comparison(feature_importance)

        # 7. Resumen numérico
        print("\nRESUMEN NUMÉRICO DE RESULTADOS")
        print("="*60)
        results_df = pd.DataFrame(metrics_dict).T
        print(results_df.round(4))

        if save_report:
            results_df.to_csv(f'csv/improvement/comparison_summary_{dataset_name.lower()}.csv')
            print(f"\nResumen guardado en: images/improvement/comparison_summary_{dataset_name.lower()}.csv")

        print(f"\nReporte completo generado exitosamente!")
        return results_df



def generate_comparison_report(hybrid_classifier, test_pairs, test_labels,
                               val_pairs=None, val_labels=None):
    """
    Generates a detailed comparison report for a hybrid classifier's test and validation
    results, including feature importance analysis and visualizations.

    The function evaluates the hybrid classifier's predictive performance on both
    testing and optional validation datasets using provided data pairs and their
    corresponding labels. It calculates predicted values and probabilities, then
    creates comprehensive comparison reports for each dataset. For the testing dataset,
    it also analyzes and incorporates feature importance details into the report.

    :param hybrid_classifier: The hybrid classifier model to evaluate. The classifier
        must implement the `predict`, `predict_proba`, and `analyze_feature_importance`
        methods.
    :type hybrid_classifier: Classifier
    :param test_pairs: The testing dataset comprising pairs of input data.
    :type test_pairs: Any
    :param test_labels: The ground-truth labels for the testing dataset.
    :type test_labels: Any
    :param val_pairs: The optional validation dataset comprising pairs of input data.
        If None, validation evaluation will not be performed.
    :type val_pairs: Any, optional
    :param val_labels: The ground-truth labels for the validation dataset. This must
        be provided if val_pairs is not None.
    :type val_labels: Any, optional
    :return: A summary dataframe containing model performance statistics, feature
        importance details, and comprehensive evaluation results for the testing dataset.
    :rtype: pandas.DataFrame
    """
    viz_tools = PlagiarismVisualizationTools()

    # Resultados del modelo híbrido
    test_pred = hybrid_classifier.predict(test_pairs)
    test_proba = hybrid_classifier.predict_proba(test_pairs)[:, 1]

    test_results = {
        'Hybrid (TF-IDF + AST)': {
            'y_true': test_labels,
            'y_pred': test_pred,
            'y_proba': test_proba
        }
    }

    # Si tenemos datos de validación, también los incluimos
    if val_pairs is not None and val_labels is not None:
        val_pred = hybrid_classifier.predict(val_pairs)
        val_proba = hybrid_classifier.predict_proba(val_pairs)[:, 1]

        val_results = {
            'Hybrid (TF-IDF + AST)': {
                'y_true': val_labels,
                'y_pred': val_pred,
                'y_proba': val_proba
            }
        }

        print("VALIDATION RESULTS:")
        viz_tools.create_comprehensive_comparison_report(val_results, dataset_name="Validation")

    print("\nTEST RESULTS:")
    feature_importance = hybrid_classifier.analyze_feature_importance(top_n=30)


    summary_df = viz_tools.create_comprehensive_comparison_report(
        test_results,
        feature_importance=feature_importance,
        dataset_name="Test"
    )

    return summary_df
