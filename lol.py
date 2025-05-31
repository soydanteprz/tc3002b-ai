#!/usr/bin/env python3
"""
Corrección específica para ConPlag con estructura version2/sub1_sub2/
"""

import shutil
from pathlib import Path
import pandas as pd
import itertools
import json

def analyze_conplag_structure(conplag_path: str):
    """
    Analiza la estructura real de ConPlag para confirmar formato
    """

    conplag_path = Path(conplag_path)
    print(f"🔍 Analizando estructura de ConPlag...")

    # Verificar estructura general
    versions_dir = conplag_path / "versions"
    if not versions_dir.exists():
        print(f"❌ No se encontró directorio versions/ en {conplag_path}")
        return False

    # Analizar versiones disponibles
    version_dirs = [d for d in versions_dir.iterdir() if d.is_dir()]
    print(f"📂 Directorios en versions/: {[d.name for d in version_dirs]}")

    # Analizar version2 específicamente
    version2_dir = versions_dir / "version_2"
    if not version2_dir.exists():
        print(f"❌ No se encontró directorio version2/")
        return False

    # Analizar contenido de version2
    pair_dirs = [d for d in version2_dir.iterdir() if d.is_dir()]
    print(f"📁 Directorios en version2/: {len(pair_dirs)}")

    if not pair_dirs:
        print(f"❌ No hay directorios en version2/")
        return False

    # Analizar estructura de pares
    print(f"\n📊 Analizando estructura de pares (muestra de 5):")

    valid_pairs = 0
    for i, pair_dir in enumerate(sorted(pair_dirs)[:5]):
        print(f"  📁 {pair_dir.name}/")

        # Verificar formato sub1_sub2
        if '_' not in pair_dir.name:
            print(f"    ❌ Formato incorrecto (esperado: sub1_sub2)")
            continue

        try:
            sub1_id, sub2_id = pair_dir.name.split('_', 1)
            print(f"    🔍 sub1: {sub1_id}, sub2: {sub2_id}")
        except:
            print(f"    ❌ Error al dividir nombre del directorio")
            continue

        # Verificar archivos Java
        java_files = list(pair_dir.glob("*.java"))
        print(f"    📄 Archivos Java: {len(java_files)}")

        expected_files = [f"{sub1_id}.java", f"{sub2_id}.java"]
        found_files = [f.name for f in java_files]

        for expected in expected_files:
            if expected in found_files:
                print(f"      ✅ {expected}")
            else:
                print(f"      ❌ {expected} (faltante)")

        if all(ef in found_files for ef in expected_files):
            valid_pairs += 1

    print(f"\n📊 Resumen: {valid_pairs}/5 pares válidos en muestra")

    if valid_pairs > 0:
        print(f"✅ Estructura ConPlag confirmada")
        return True
    else:
        print(f"❌ Estructura ConPlag no válida")
        return False

def build_conplag_index(conplag_path: str):
    """
    Construye índice de archivos ConPlag con estructura corregida
    """

    conplag_path = Path(conplag_path)
    print(f"🔍 Construyendo índice ConPlag...")

    java_files_index = {}

    # Buscar en version2 con estructura sub1_sub2
    version2_dir = conplag_path / "versions" / "version2"

    if version2_dir.exists():
        for pair_dir in version2_dir.iterdir():
            if pair_dir.is_dir() and '_' in pair_dir.name:
                try:
                    sub1_id, sub2_id = pair_dir.name.split('_', 1)

                    # Buscar archivos específicos
                    sub1_file = pair_dir / f"{sub1_id}.java"
                    sub2_file = pair_dir / f"{sub2_id}.java"

                    if sub1_file.exists():
                        java_files_index[sub1_id] = sub1_file
                    if sub2_file.exists():
                        java_files_index[sub2_id] = sub2_file

                except Exception as e:
                    print(f"    ⚠️ Error procesando {pair_dir.name}: {e}")

    print(f"✅ Índice construido: {len(java_files_index)} archivos")
    return java_files_index

def verify_labels_connection(conplag_path: str, java_files_index: dict):
    """
    Verifica conexión entre labels.csv y archivos indexados
    """

    conplag_path = Path(conplag_path)
    labels_file = conplag_path / "versions" / "labels.csv"

    if not labels_file.exists():
        print(f"❌ No se encontró labels.csv")
        return False

    labels_df = pd.read_csv(labels_file)
    print(f"📊 Verificando {len(labels_df)} pares de labels.csv...")

    found_pairs = 0
    missing_files = []

    # Verificar muestra de 20 pares
    sample_df = labels_df.head(20)

    for _, row in sample_df.iterrows():
        sub1_id = row['sub1']
        sub2_id = row['sub2']

        sub1_found = sub1_id in java_files_index
        sub2_found = sub2_id in java_files_index

        if sub1_found and sub2_found:
            found_pairs += 1
        else:
            if not sub1_found:
                missing_files.append(sub1_id)
            if not sub2_found:
                missing_files.append(sub2_id)

    print(f"✅ {found_pairs}/{len(sample_df)} pares verificados correctamente")

    if missing_files:
        unique_missing = list(set(missing_files))
        print(f"⚠️ IDs no encontrados: {unique_missing[:5]}")
        if len(unique_missing) > 5:
            print(f"   ... y {len(unique_missing) - 5} más")

    return found_pairs > len(sample_df) * 0.8  # 80% éxito mínimo

def rebuild_conplag_cases(conplag_path: str, unified_dataset_path: str):
    """
    Reconstruye casos de ConPlag con estructura corregida
    """

    conplag_path = Path(conplag_path)
    unified_dataset_path = Path(unified_dataset_path)

    print(f"🔄 Reconstruyendo casos ConPlag...")

    # Construir índice corregido
    java_files_index = build_conplag_index(conplag_path)

    if len(java_files_index) == 0:
        print(f"❌ No se pudieron indexar archivos ConPlag")
        return False

    # Verificar conexión con labels
    if not verify_labels_connection(conplag_path, java_files_index):
        print(f"❌ La verificación de labels falló")
        return False

    # Cargar labels
    labels_file = conplag_path / "versions" / "labels.csv"
    labels_df = pd.read_csv(labels_file)

    # Agrupar por problema
    problems = labels_df['problem'].unique()
    print(f"📋 Reconstruyendo {len(problems)} problemas...")

    # Eliminar casos ConPlag existentes
    contest_cases = [d for d in unified_dataset_path.glob("Contest-*") if d.is_dir()]
    for case_dir in contest_cases:
        print(f"🗑️ Eliminando {case_dir.name} existente...")
        shutil.rmtree(case_dir)

    # Reconstruir casos
    total_files_copied = 0
    rebuilt_cases = {}

    for case_num, problem_id in enumerate(sorted(problems), 1):
        case_name = f"Contest-{case_num:02d}"
        case_dir = unified_dataset_path / case_name
        case_dir.mkdir(exist_ok=True)

        print(f"📁 Creando {case_name} (Problema {problem_id})...")

        # Crear estructura
        original_dir = case_dir / "codigo_original"
        no_plagio_dir = case_dir / "no_plagio"
        plagio_dir = case_dir / "plagio"

        original_dir.mkdir(exist_ok=True)
        no_plagio_dir.mkdir(exist_ok=True)
        plagio_dir.mkdir(exist_ok=True)
        (plagio_dir / "contest_level").mkdir(exist_ok=True)

        # Filtrar pares del problema
        problem_pairs = labels_df[labels_df['problem'] == problem_id]

        file_counter = 1
        plagio_version = 1
        no_plagio_version = 1
        files_copied_this_case = 0

        for _, pair in problem_pairs.iterrows():
            sub1_id = pair['sub1']
            sub2_id = pair['sub2']
            verdict = pair['verdict']

            # Buscar archivos en índice
            if sub1_id in java_files_index and sub2_id in java_files_index:
                file1_path = java_files_index[sub1_id]
                file2_path = java_files_index[sub2_id]

                try:
                    if verdict == 1:  # Plagiado
                        # Original
                        dest1 = original_dir / f"original_{file_counter}.java"
                        shutil.copy2(file1_path, dest1)

                        # Plagio
                        version_dir = plagio_dir / "contest_level" / f"version_{plagio_version:02d}"
                        version_dir.mkdir(exist_ok=True)
                        dest2 = version_dir / f"plagiarized_{file_counter}.java"
                        shutil.copy2(file2_path, dest2)

                        plagio_version += 1
                        files_copied_this_case += 2

                    else:  # No plagiado
                        # Ambos como independientes
                        version_dir1 = no_plagio_dir / f"version_{no_plagio_version:02d}"
                        version_dir1.mkdir(exist_ok=True)
                        dest1 = version_dir1 / f"independent_{file_counter}a.java"
                        shutil.copy2(file1_path, dest1)
                        no_plagio_version += 1

                        version_dir2 = no_plagio_dir / f"version_{no_plagio_version:02d}"
                        version_dir2.mkdir(exist_ok=True)
                        dest2 = version_dir2 / f"independent_{file_counter}b.java"
                        shutil.copy2(file2_path, dest2)
                        no_plagio_version += 1

                        files_copied_this_case += 2

                    file_counter += 1

                except Exception as e:
                    print(f"    ⚠️ Error copiando {sub1_id}-{sub2_id}: {e}")

        total_files_copied += files_copied_this_case

        # Registrar caso
        rebuilt_cases[case_name] = {
            'problem_id': problem_id,
            'pairs_processed': len(problem_pairs),
            'files_copied': files_copied_this_case,
            'original_files': len(list(original_dir.glob("*.java"))),
            'no_plagio_versions': len(list(no_plagio_dir.glob("version_*"))),
            'plagio_levels': 1
        }

        print(f"    ✅ {files_copied_this_case} archivos copiados")

    print(f"✅ Reconstrucción ConPlag completada!")
    print(f"📊 {len(rebuilt_cases)} casos reconstruidos")
    print(f"📄 {total_files_copied} archivos copiados")

    return rebuilt_cases

def regenerate_unified_pairs(unified_dataset_path: str):
    """
    Regenera todos los pares del dataset unificado
    """

    unified_dataset_path = Path(unified_dataset_path)
    print(f"🔄 Regenerando pares del dataset unificado...")

    all_pairs = []
    pair_id = 1

    # Procesar todos los casos (IR-Plag + ConPlag reconstruido)
    all_cases = sorted([d for d in unified_dataset_path.glob("*")
                        if d.is_dir() and (d.name.startswith(("case-", "Contest-")))])

    for case_dir in all_cases:
        case_name = case_dir.name
        source = "ir_plag" if case_name.startswith("case-") else "conplag"

        print(f"  🔄 Procesando {case_name}...")

        # Obtener archivos
        original_files = list((case_dir / "codigo_original").glob("*.java"))
        no_plagio_files = []
        plagio_files = []

        # No plagio
        no_plagio_dir = case_dir / "no_plagio"
        for version_dir in no_plagio_dir.glob("version_*"):
            no_plagio_files.extend(list(version_dir.glob("*.java")))

        # Plagio
        plagio_dir = case_dir / "plagio"
        for level_dir in plagio_dir.iterdir():
            if level_dir.is_dir():
                level_name = level_dir.name
                for version_dir in level_dir.glob("version_*"):
                    for java_file in version_dir.glob("*.java"):
                        plagio_files.append((java_file, level_name))

        # Generar pares no plagiados
        for orig_file in original_files:
            for no_plag_file in no_plagio_files:
                all_pairs.append({
                    'pair_id': f"pair_{pair_id:06d}",
                    'case': case_name,
                    'file1': str(orig_file.relative_to(unified_dataset_path)),
                    'file2': str(no_plag_file.relative_to(unified_dataset_path)),
                    'label': 0,
                    'plagiarism_level': 'none',
                    'source_dataset': source,
                    'comparison_type': 'original_vs_independent'
                })
                pair_id += 1

        for file1, file2 in itertools.combinations(no_plagio_files, 2):
            all_pairs.append({
                'pair_id': f"pair_{pair_id:06d}",
                'case': case_name,
                'file1': str(file1.relative_to(unified_dataset_path)),
                'file2': str(file2.relative_to(unified_dataset_path)),
                'label': 0,
                'plagiarism_level': 'none',
                'source_dataset': source,
                'comparison_type': 'independent_vs_independent'
            })
            pair_id += 1

        # Generar pares plagiados
        for orig_file in original_files:
            for plag_file, level in plagio_files:
                all_pairs.append({
                    'pair_id': f"pair_{pair_id:06d}",
                    'case': case_name,
                    'file1': str(orig_file.relative_to(unified_dataset_path)),
                    'file2': str(plag_file.relative_to(unified_dataset_path)),
                    'label': 1,
                    'plagiarism_level': level,
                    'source_dataset': source,
                    'comparison_type': 'original_vs_plagiarized'
                })
                pair_id += 1

        # Plagiado vs plagiado
        files_by_level = {}
        for plag_file, level in plagio_files:
            if level not in files_by_level:
                files_by_level[level] = []
            files_by_level[level].append(plag_file)

        for level, level_files in files_by_level.items():
            for file1, file2 in itertools.combinations(level_files, 2):
                all_pairs.append({
                    'pair_id': f"pair_{pair_id:06d}",
                    'case': case_name,
                    'file1': str(file1.relative_to(unified_dataset_path)),
                    'file2': str(file2.relative_to(unified_dataset_path)),
                    'label': 1,
                    'plagiarism_level': level,
                    'source_dataset': source,
                    'comparison_type': 'plagiarized_vs_plagiarized'
                })
                pair_id += 1

    # Guardar pares
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(unified_dataset_path / "all_pairs.csv", index=False)

    # Crear splits
    pairs_shuffled = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(pairs_shuffled))

    train_df = pairs_shuffled[:train_size]
    test_df = pairs_shuffled[train_size:]

    train_df.to_csv(unified_dataset_path / "train_pairs.csv", index=False)
    test_df.to_csv(unified_dataset_path / "test_pairs.csv", index=False)

    # Estadísticas
    plagiarized = len(pairs_df[pairs_df['label'] == 1])
    print(f"✅ Dataset regenerado: {len(pairs_df)} pares totales")
    print(f"  - Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  - Plagiados: {plagiarized} ({plagiarized/len(pairs_df)*100:.1f}%)")

    return pairs_df

def main():
    # Configurar rutas
    conplag_path = "data/conplag"  # Ajustar
    unified_dataset_path = "data/final_dataset"  # Ajustar

    print("🔧 CORRECCIÓN ESPECÍFICA DE CONPLAG")
    print("=" * 40)

    # 1. Analizar estructura
    if not analyze_conplag_structure(conplag_path):
        print("❌ La estructura de ConPlag no es válida")
        return False

    # 2. Reconstruir casos ConPlag
    rebuilt_cases = rebuild_conplag_cases(conplag_path, unified_dataset_path)

    if not rebuilt_cases:
        print("❌ La reconstrucción de ConPlag falló")
        return False

    # 3. Regenerar todos los pares
    pairs_df = regenerate_unified_pairs(unified_dataset_path)

    print(f"\n🎉 CORRECCIÓN CONPLAG COMPLETADA!")
    print(f"📊 Dataset final optimizado")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Ejecuta el analizador para ver las estadísticas actualizadas")
    else:
        print("❌ La corrección falló, revisa los logs")
