import warnings
warnings.filterwarnings('ignore')

from plagiarism_ast_detector import EnhancedPlagiarismDetector
from plagiarism_analyzer import PlagiarismAnalyzer
import sys
import os


def main():
    """Función principal con salida limpia"""
    # Configuración
    BASE_PATH = 'data/splits/train'
    CSV_PATH = 'data/splits/train.csv'
    OUTPUT_CSV = 'similarity_ast_enhanced.csv'

    # Verificar que existen los archivos
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encontró {CSV_PATH}")
        sys.exit(1)

    print("DETECTOR DE PLAGIO CON NORMALIZACIÓN AST")
    print("=" * 60)

    # 1. Ejecutar detección (sin mensajes de progreso)
    detector = EnhancedPlagiarismDetector()
    problematic_files = detector.find_problematic_files(BASE_PATH, CSV_PATH)

    # To debug a specific file
    if len(problematic_files) > 0:
        print("\n🔬 Debugging first problematic file in detail:")
        detector.debug_file_parsing(problematic_files[0][0])
    df_results = detector.process_dataset(BASE_PATH, CSV_PATH, OUTPUT_CSV, verbose=True)

    # 2. Análisis de métricas individuales
    print("\n🔍 Análisis detallado de métricas:")
    print("-" * 60)
    metrics_cols = ['tfidf_normalized', 'ast_structure', 'method_signatures', 'sequence_normalized', 'combined_score']

    for metric in metrics_cols:
        plag_mean = df_results[df_results['es_plagio']==1][metric].mean()
        no_plag_mean = df_results[df_results['es_plagio']==0][metric].mean()
        print(f"{metric}:")
        print(f"  Plagio=1: {plag_mean:.3f}")
        print(f"  Plagio=0: {no_plag_mean:.3f}")

    # 3. Análisis con visualización (opcional)
    if os.path.exists(OUTPUT_CSV):
        analyzer = PlagiarismAnalyzer(OUTPUT_CSV)

        # Encontrar mejor umbral
        print("\n" + "="*60)
        best_threshold = analyzer.evaluate_thresholds('combined_score')

        # Generar reporte
        analyzer.generate_report(threshold=best_threshold, metric='combined_score')

        # Guardar gráficos principales
        print("\n📊 Generando visualizaciones...")
        analyzer.plot_metrics_distribution()
        analyzer.plot_roc_curves()
        print("✅ Visualizaciones guardadas")

    print("\n✅ Proceso completado exitosamente!")

def create_hybrid_detector(comparison_csv="comparison_results.csv", output_csv="hybrid_results.csv"):
    """
    Creates a hybrid plagiarism detector combining TF-IDF and AST-CC scores.

    Args:
        comparison_csv: CSV with both TF-IDF and AST-CC results
        output_csv: Path to save the hybrid model results
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Load comparison results
    print("🔄 Creating hybrid plagiarism detector...")
    df = pd.read_csv(comparison_csv)

    # Create hybrid models
    df['weighted_avg'] = 0.7 * df['tfidf_similarity'] + 0.3 * df['astcc_similarity']
    df['max_score'] = np.maximum(df['tfidf_similarity'], df['astcc_similarity'])
    df['voting'] = ((df['tfidf_prediction'] + df['astcc_prediction']) >= 1).astype(int)

    # Find optimal threshold for weighted average
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0.3, 0.8, 20)
    results = []

    print("🔍 Finding optimal threshold...")
    for threshold in thresholds:
        pred = (df['weighted_avg'] >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['label'], pred, average='binary', zero_division=0)
        accuracy = accuracy_score(df['label'], pred)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Create hybrid predictions using best threshold
    df['hybrid_weighted_pred'] = (df['weighted_avg'] >= best_threshold).astype(int)
    df['hybrid_weighted_correct'] = (df['hybrid_weighted_pred'] == df['label'])

    # Add optimal max score threshold
    max_results = []
    for threshold in thresholds:
        pred = (df['max_score'] >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['label'], pred, average='binary', zero_division=0)
        max_results.append({'threshold': threshold, 'f1': f1})

    best_max_threshold = pd.DataFrame(max_results).loc[pd.DataFrame(max_results)['f1'].idxmax(), 'threshold']
    df['hybrid_max_pred'] = (df['max_score'] >= best_max_threshold).astype(int)
    df['hybrid_max_correct'] = (df['hybrid_max_pred'] == df['label'])

    # Calculate accuracy for each method
    tfidf_acc = df['tfidf_correct'].mean()
    astcc_acc = df['astcc_correct'].mean()
    weighted_acc = df['hybrid_weighted_correct'].mean()
    max_acc = df['hybrid_max_correct'].mean()
    voting_acc = (df['voting'] == df['label']).mean()

    # Save hybrid results
    df.to_csv(output_csv, index=False)

    # Print comparison
    print("\n📊 Hybrid Model Performance:")
    print(f"  TF-IDF Accuracy:        {tfidf_acc:.2%}")
    print(f"  AST-CC Accuracy:        {astcc_acc:.2%}")
    print(f"  Weighted Avg Accuracy:  {weighted_acc:.2%} (threshold: {best_threshold:.2f})")
    print(f"  Max Score Accuracy:     {max_acc:.2%} (threshold: {best_max_threshold:.2f})")
    print(f"  Simple Voting Accuracy: {voting_acc:.2%}")

    # Plot results comparison
    plt.figure(figsize=(10, 6))
    methods = ['TF-IDF', 'AST-CC', 'Weighted\nAvg', 'Max\nScore', 'Voting']
    accuracies = [tfidf_acc, astcc_acc, weighted_acc, max_acc, voting_acc]

    plt.bar(methods, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.axhline(y=tfidf_acc, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Plagiarism Detection Methods Comparison')
    plt.ylim(0, max(accuracies) * 1.1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.1%}", ha='center')

    plt.tight_layout()
    plt.savefig('hybrid_comparison.png', dpi=300)

    print(f"\n💾 Results saved to {output_csv}")
    print(f"📊 Visual comparison saved to hybrid_comparison.png")

    # Return best method
    methods_dict = {
        'tfidf': tfidf_acc,
        'astcc': astcc_acc,
        'weighted': weighted_acc,
        'max': max_acc,
        'voting': voting_acc
    }
    best_method = max(methods_dict, key=methods_dict.get)
    best_accuracy = methods_dict[best_method]
    print(f"\nBest method: {best_method.upper()} with {best_accuracy:.2%} accuracy")

    return df

if __name__ == "__main__":

    # Then create hybrid model
    create_hybrid_detector()
