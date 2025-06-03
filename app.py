#!/usr/bin/env python
"""
Script principal optimizado para detección de plagio con salida limpia
"""

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
        print(f"❌ Error: No se encontró {CSV_PATH}")
        sys.exit(1)

    print("🚀 DETECTOR DE PLAGIO CON NORMALIZACIÓN AST")
    print("=" * 60)

    # 1. Ejecutar detección (sin mensajes de progreso)
    detector = EnhancedPlagiarismDetector()
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


if __name__ == '__main__':
    main()
