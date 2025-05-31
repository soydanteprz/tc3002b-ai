#!/usr/bin/env python3
"""
Script de debug para entender la estructura de ConPlag
"""

import pandas as pd
from pathlib import Path
import os

def debug_conplag_structure(conplag_path: str):
    """Explora la estructura de ConPlag para entender el formato"""

    conplag_path = Path(conplag_path)
    print(f"🔍 Explorando estructura de ConPlag en: {conplag_path}")

    # 1. Verificar estructura general
    print("\n📁 ESTRUCTURA GENERAL:")
    for item in conplag_path.iterdir():
        if item.is_dir():
            print(f"  📂 {item.name}/")
            # Contar archivos Java en subdirectorios
            java_count = len(list(item.rglob("*.java")))
            if java_count > 0:
                print(f"    └── {java_count} archivos .java")
        else:
            print(f"  📄 {item.name}")

    # 2. Examinar labels.csv
    labels_file = conplag_path / "versions" / "labels.csv"
    if labels_file.exists():
        print(f"\n📊 ANÁLISIS DE labels.csv:")
        labels_df = pd.read_csv(labels_file)

        print(f"  Columnas: {list(labels_df.columns)}")
        print(f"  Filas: {len(labels_df)}")
        print(f"\n  Primeras 5 filas:")
        print(labels_df.head())

        # Estadísticas por columna
        for col in labels_df.columns:
            if labels_df[col].dtype in ['int64', 'float64']:
                print(f"\n  {col}:")
                print(f"    Valores únicos: {labels_df[col].nunique()}")
                print(f"    Rango: {labels_df[col].min()} - {labels_df[col].max()}")
                if col == 'verdict':
                    print(f"    Distribución: {labels_df[col].value_counts().to_dict()}")
            else:
                print(f"\n  {col}:")
                print(f"    Valores únicos: {labels_df[col].nunique()}")
                print(f"    Ejemplos: {labels_df[col].unique()[:5]}")

    # 3. Buscar archivos de código
    print(f"\n🔍 BÚSQUEDA DE ARCHIVOS DE CÓDIGO:")

    versions_dir = conplag_path / "versions"
    if versions_dir.exists():
        print("  Explorando directorio versions/:")

        for version_subdir in versions_dir.iterdir():
            if version_subdir.is_dir():
                java_files = list(version_subdir.rglob("*.java"))
                print(f"    📂 {version_subdir.name}: {len(java_files)} archivos .java")

                # Mostrar algunos ejemplos de nombres
                if java_files:
                    print("      Ejemplos de archivos:")
                    for java_file in java_files[:5]:
                        rel_path = java_file.relative_to(version_subdir)
                        print(f"        - {rel_path}")
                    if len(java_files) > 5:
                        print(f"        ... y {len(java_files) - 5} más")

    # 4. Verificar si existen archivos con IDs de submission
    if labels_file.exists():
        print(f"\n🔗 VERIFICACIÓN DE ENLACES SUBMISSION -> ARCHIVO:")
        labels_df = pd.read_csv(labels_file)

        # Tomar algunas submissions de muestra
        sample_subs = []
        if 'sub1' in labels_df.columns:
            sample_subs.extend(labels_df['sub1'].head(3).tolist())
        if 'sub2' in labels_df.columns:
            sample_subs.extend(labels_df['sub2'].head(3).tolist())

        for sub_id in sample_subs:
            print(f"\n  Buscando archivos para submission {sub_id}:")

            # Posibles nombres de archivo
            possible_names = [
                f"{sub_id}.java",
                f"{sub_id}",
                sub_id
            ]

            found = False
            for possible_name in possible_names:
                # Buscar en toda la estructura
                matches = list(conplag_path.rglob(possible_name))
                if matches:
                    print(f"    ✅ Encontrado como '{possible_name}':")
                    for match in matches[:3]:  # Mostrar máximo 3
                        rel_path = match.relative_to(conplag_path)
                        print(f"      - {rel_path}")
                    found = True
                    break

            if not found:
                print(f"    ❌ No encontrado con nombres: {possible_names}")

    # 5. Análisis de estructura de archivos
    print(f"\n📈 RESUMEN DE ARCHIVOS:")
    total_java = len(list(conplag_path.rglob("*.java")))
    print(f"  Total archivos .java: {total_java}")

    # Buscar patrones en nombres de directorio
    all_dirs = [d for d in conplag_path.rglob("*") if d.is_dir()]
    print(f"  Total directorios: {len(all_dirs)}")

    return labels_df if labels_file.exists() else None

def suggest_fixes(conplag_path: str):
    """Sugiere correcciones basadas en la estructura encontrada"""
    print(f"\n💡 SUGERENCIAS DE CORRECCIÓN:")

    conplag_path = Path(conplag_path)

    # Buscar archivos Java y analizar patrones de nombres
    java_files = list(conplag_path.rglob("*.java"))

    if java_files:
        print(f"  1. Patrón de nombres de archivos:")

        # Analizar nombres de archivos
        stems = [f.stem for f in java_files[:10]]
        print(f"     Ejemplos de nombres: {stems}")

        # Ver si siguen algún patrón hexadecimal como los IDs
        hex_pattern_files = [f for f in java_files if len(f.stem) == 8 and all(c in '0123456789abcdef' for c in f.stem.lower())]

        if hex_pattern_files:
            print(f"  2. ✅ Encontrados {len(hex_pattern_files)} archivos con patrón hexadecimal (8 chars)")
            print(f"     Esto coincide con el formato de sub1/sub2 en labels.csv")
            print(f"     Ubicación típica: {hex_pattern_files[0].parent.relative_to(conplag_path)}")
        else:
            print(f"  2. ❌ No se encontraron archivos con patrón hexadecimal")
            print(f"     Los archivos podrían estar en una estructura diferente")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python debug_conplag.py <ruta_a_conplag>")
        sys.exit(1)

    conplag_path = sys.argv[1]

    if not Path(conplag_path).exists():
        print(f"❌ Ruta no existe: {conplag_path}")
        sys.exit(1)

    labels_df = debug_conplag_structure(conplag_path)
    suggest_fixes(conplag_path)

    print(f"\n✅ Debug completado. Usa esta información para ajustar el unificador.")
