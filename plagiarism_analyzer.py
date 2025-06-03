import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import numpy as np

class PlagiarismAnalyzer:
    """Analiza y visualiza los resultados del detector de plagio"""

    def __init__(self, results_csv='similarity_ast_enhanced.csv'):
        self.df = pd.read_csv(results_csv)
        self.metrics = ['tfidf_normalized', 'ast_structure', 'method_signatures',
                        'sequence_normalized', 'combined_score']

    def plot_metrics_distribution(self):
        """Visualiza la distribución de cada métrica por tipo (plagio/no plagio)"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]

            # Datos para plagio y no plagio
            plagiarized = self.df[self.df['es_plagio'] == 1][metric]
            not_plagiarized = self.df[self.df['es_plagio'] == 0][metric]

            # Histogramas
            ax.hist(not_plagiarized, bins=30, alpha=0.5, label='No Plagio', color='blue')
            ax.hist(plagiarized, bins=30, alpha=0.5, label='Plagio', color='red')

            ax.set_xlabel(metric)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de {metric}')
            ax.legend()

            # Agregar líneas de media
            ax.axvline(not_plagiarized.mean(), color='blue', linestyle='--', linewidth=2)
            ax.axvline(plagiarized.mean(), color='red', linestyle='--', linewidth=2)

        # Ocultar el sexto subplot vacío
        axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig('metrics_distribution.png', dpi=300)
        plt.show()

    def plot_correlation_matrix(self):
        """Visualiza la correlación entre métricas"""
        correlation_matrix = self.df[self.metrics].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": .8})
        plt.title('Correlación entre Métricas de Similitud')
        plt.tight_layout()
        plt.savefig('metrics_correlation.png', dpi=300)
        plt.show()

    def evaluate_thresholds(self, metric='combined_score'):
        """Evalúa diferentes umbrales para una métrica"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []

        for threshold in thresholds:
            predictions = (self.df[metric] >= threshold).astype(int)

            # Calcular métricas
            tp = ((predictions == 1) & (self.df['es_plagio'] == 1)).sum()
            tn = ((predictions == 0) & (self.df['es_plagio'] == 0)).sum()
            fp = ((predictions == 1) & (self.df['es_plagio'] == 0)).sum()
            fn = ((predictions == 0) & (self.df['es_plagio'] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(self.df)

            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })

        results_df = pd.DataFrame(results)

        # Visualizar
        plt.figure(figsize=(12, 8))
        plt.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='o')
        plt.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='s')
        plt.plot(results_df['threshold'], results_df['f1'], label='F1-Score', marker='^')
        plt.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', marker='d')

        plt.xlabel('Umbral')
        plt.ylabel('Puntuación')
        plt.title(f'Métricas de Evaluación vs Umbral ({metric})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'threshold_evaluation_{metric}.png', dpi=300)
        plt.show()

        # Encontrar mejor umbral
        best_idx = results_df['f1'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        print(f"\nMejor umbral para {metric}: {best_threshold:.3f}")
        print(f"   F1-Score: {results_df.loc[best_idx, 'f1']:.3f}")
        print(f"   Precision: {results_df.loc[best_idx, 'precision']:.3f}")
        print(f"   Recall: {results_df.loc[best_idx, 'recall']:.3f}")

        return best_threshold

    def plot_roc_curves(self):
        """Visualiza las curvas ROC para cada métrica"""
        plt.figure(figsize=(10, 8))

        for metric in self.metrics:
            fpr, tpr, _ = roc_curve(self.df['es_plagio'], self.df[metric])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{metric} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Comparación de Métricas')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300)
        plt.show()

    def analyze_failures(self, threshold=0.5, metric='combined_score'):
        """Analiza casos de fallo (falsos positivos y negativos)"""
        predictions = (self.df[metric] >= threshold).astype(int)

        # Falsos positivos
        fp_mask = (predictions == 1) & (self.df['es_plagio'] == 0)
        fp_cases = self.df[fp_mask].sort_values(metric, ascending=False)

        # Falsos negativos
        fn_mask = (predictions == 0) & (self.df['es_plagio'] == 1)
        fn_cases = self.df[fn_mask].sort_values(metric, ascending=True)

        print("\nAnálisis de Fallos")
        print("=" * 60)

        print(f"\nFalsos Positivos ({len(fp_cases)} casos):")
        print("  (Archivos NO plagiados pero detectados como plagio)")
        if len(fp_cases) > 0:
            print("\n  Top 5 casos más problemáticos:")
            display_cols = ['folder', 'dataset'] + self.metrics
            print(fp_cases[display_cols].head().to_string(index=False))

            # Análisis de patrones
            print(f"\n  Distribución por dataset:")
            print(f"    • conplag: {(fp_cases['dataset'] == 'conplag').sum()}")
            print(f"    • ir_plag: {(fp_cases['dataset'] == 'ir_plag').sum()}")

        print(f"\n\nFalsos Negativos ({len(fn_cases)} casos):")
        print("  (Archivos plagiados NO detectados)")
        if len(fn_cases) > 0:
            print("\n  Top 5 casos más problemáticos:")
            print(fn_cases[display_cols].head().to_string(index=False))

            # Análisis de patrones
            print(f"\n  Distribución por dataset:")
            print(f"    • conplag: {(fn_cases['dataset'] == 'conplag').sum()}")
            print(f"    • ir_plag: {(fn_cases['dataset'] == 'ir_plag').sum()}")

        # Análisis de características de los fallos
        if len(fp_cases) > 0 or len(fn_cases) > 0:
            print("\n\nCaracterísticas de los casos problemáticos:")

            if len(fp_cases) > 0:
                print("\n  Falsos Positivos - Valores promedio:")
                for metric_name in self.metrics:
                    avg = fp_cases[metric_name].mean()
                    print(f"    • {metric_name}: {avg:.3f}")

            if len(fn_cases) > 0:
                print("\n  Falsos Negativos - Valores promedio:")
                for metric_name in self.metrics:
                    avg = fn_cases[metric_name].mean()
                    print(f"    • {metric_name}: {avg:.3f}")

        return fp_cases, fn_cases

    def generate_report(self, threshold=0.5, metric='combined_score'):
        """Genera un reporte completo de evaluación"""
        predictions = (self.df[metric] >= threshold).astype(int)

        print("\nREPORTE DE EVALUACIÓN DEL DETECTOR DE PLAGIO")
        print("=" * 60)

        # Reporte de clasificación
        print("\nReporte de Clasificación:")
        print(classification_report(self.df['es_plagio'], predictions,
                                    target_names=['No Plagio', 'Plagio']))

        # Matriz de confusión
        cm = confusion_matrix(self.df['es_plagio'], predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Plagio', 'Plagio'],
                    yticklabels=['No Plagio', 'Plagio'])
        plt.title(f'Matriz de Confusión ({metric}, umbral={threshold})')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()

        # Estadísticas por dataset
        print("\nEstadísticas por Dataset:")
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            dataset_pred = (dataset_df[metric] >= threshold).astype(int)

            accuracy = (dataset_pred == dataset_df['es_plagio']).mean()
            print(f"\n{dataset}:")
            print(f"  Muestras: {len(dataset_df)}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Plagio real: {dataset_df['es_plagio'].sum()}")
            print(f"  Plagio detectado: {dataset_pred.sum()}")

    def compare_individual_metrics(self):
        """Compara el rendimiento individual de cada métrica"""
        results = []

        for metric in self.metrics:
            # Encontrar mejor umbral para cada métrica
            best_threshold = self.evaluate_thresholds(metric)
            predictions = (self.df[metric] >= best_threshold).astype(int)

            # Calcular métricas
            tp = ((predictions == 1) & (self.df['es_plagio'] == 1)).sum()
            tn = ((predictions == 0) & (self.df['es_plagio'] == 0)).sum()
            fp = ((predictions == 1) & (self.df['es_plagio'] == 0)).sum()
            fn = ((predictions == 0) & (self.df['es_plagio'] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(self.df)

            # AUC
            fpr, tpr, _ = roc_curve(self.df['es_plagio'], self.df[metric])
            roc_auc = auc(fpr, tpr)

            results.append({
                'metric': metric,
                'best_threshold': best_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'auc': roc_auc
            })

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        print("\nComparación de Métricas Individuales:")
        print(comparison_df.to_string(index=False))

        # Visualizar comparación
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison_df))
        width = 0.2

        ax.bar(x - width*1.5, comparison_df['precision'], width, label='Precision')
        ax.bar(x - width*0.5, comparison_df['recall'], width, label='Recall')
        ax.bar(x + width*0.5, comparison_df['f1'], width, label='F1-Score')
        ax.bar(x + width*1.5, comparison_df['auc'], width, label='AUC')

        ax.set_xlabel('Métrica')
        ax.set_ylabel('Puntuación')
        ax.set_title('Comparación de Rendimiento por Métrica')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['metric'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300)
        plt.show()

        return comparison_df


# Función de ejemplo para ejecutar análisis
def run_analysis():
    """Ejecuta el análisis completo"""
    print("Análisis de Plagio resultados")

    analyzer = PlagiarismAnalyzer('similarity_ast_enhanced.csv')

    # 1. Distribución de métricas
    analyzer.plot_metrics_distribution()

    # 2. Correlación entre métricas
    analyzer.plot_correlation_matrix()

    # 3. Curvas ROC
    analyzer.plot_roc_curves()

    # 4. Evaluación de umbrales
    best_threshold = analyzer.evaluate_thresholds('combined_score')

    # 5. Reporte completo
    analyzer.generate_report(threshold=best_threshold, metric='combined_score')

    # 6. Comparación de métricas
    comparison = analyzer.compare_individual_metrics()

    best_threshold = comparison.loc[comparison['f1'].idxmax(), 'best_threshold']
    print(f"\nMejor umbral encontrado: {best_threshold:.3f} con F1-Score: {comparison['f1'].max():.3f}")


    # 7. Análisis de fallos
    print("\nAnalizando casos de fallo")
    fp_cases, fn_cases = analyzer.analyze_failures(threshold=best_threshold)

    if not fp_cases.empty or not fn_cases.empty:
        print("\nCasos de fallo analizados:")
        if not fp_cases.empty:
            print(f"  Falsos Positivos: {len(fp_cases)} casos")
        if not fn_cases.empty:
            print(f"  Falsos Negativos: {len(fn_cases)} casos")

    else:
        print("\nNo se encontraron casos de fallo significativos.")

    print("\nArchivos generados:")
    print("  - metrics_distribution.png")
    print("  - metrics_correlation.png")
    print("  - roc_curves.png")
    print("  - threshold_evaluation_combined_score.png")
    print("  - confusion_matrix.png")
    print("  - metrics_comparison.png")


if __name__ == '__main__':
    run_analysis()
