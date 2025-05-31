#!/usr/bin/env python3
"""
Convertidor unificado que procesa tanto IR-Plag como ConPlag
- IR-Plag: convierte a formato compacto (c01_orig_np01/, etc.)
- ConPlag: mueve carpetas de version_2/ al directorio unificado
- Genera CSV unificado con metadata de ambos datasets
"""

import shutil
from pathlib import Path
import pandas as pd

def convert_ir_plag_compact(ir_plag_path: str, output_path: Path):
    """
    Convierte IR-Plag a formato compacto con nombres claros
    """

    ir_plag_path = Path(ir_plag_path)

    print("🔄 Procesando IR-Plag...")
    print(f"📂 Entrada: {ir_plag_path}")

    # Buscar casos
    case_dirs = sorted([d for d in ir_plag_path.glob("case-*") if d.is_dir()])
    if not case_dirs:
        case_dirs = sorted([d for d in ir_plag_path.glob("Case-*") if d.is_dir()])

    if not case_dirs:
        print("❌ No se encontraron casos en IR-Plag")
        return []

    print(f"📁 Encontrados {len(case_dirs)} casos")

    pairs_metadata = []
    total_pairs = 0

    for case_dir in case_dirs:
        case_name = case_dir.name
        case_number = int(case_name.split('-')[1])  # case-1 -> 1
        case_id = f"c{case_number:02d}"  # c01, c02, etc.

        print(f"  📂 Procesando {case_name} → {case_id}")

        # 1. OBTENER ARCHIVO ORIGINAL
        original_dir = case_dir / "Original"
        if not original_dir.exists():
            print(f"    ❌ No se encontró directorio Original")
            continue

        original_files = list(original_dir.glob("*.java"))
        if not original_files:
            print(f"    ❌ No se encontraron archivos Java originales")
            continue

        original_file = original_files[0]

        # 2. PROCESAR NO-PLAGIADOS
        non_plag_dir = case_dir / "non-plagiarized"
        if non_plag_dir.exists():
            # Buscar directorios numerados y ordenarlos
            numeric_dirs = sorted([
                d for d in non_plag_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ], key=lambda x: int(x.name))

            np_count = 0
            for version_dir in numeric_dirs:
                java_files = list(version_dir.glob("*.java"))
                if java_files:
                    np_count += 1

                    # Crear nombre de carpeta compacto
                    folder_name = f"{case_id}_orig_np{np_count:02d}"
                    pair_folder = output_path / folder_name
                    pair_folder.mkdir(exist_ok=True)

                    # Copiar archivos con nombres estándar
                    orig_dest = pair_folder / "original.java"
                    comp_dest = pair_folder / "compared.java"

                    shutil.copy2(original_file, orig_dest)
                    shutil.copy2(java_files[0], comp_dest)

                    # Registrar metadata
                    pairs_metadata.append({
                        'folder_name': folder_name,
                        'case_id': case_id,
                        'case_orig': case_name,
                        'file1': 'original.java',
                        'file2': 'compared.java',
                        'label': 0,  # No plagiado
                        'plagiarism_level': 'none',
                        'source_dataset': 'ir_plag',
                        'comparison_type': 'original_vs_nonplagiarized'
                    })
                    total_pairs += 1

            print(f"    ✅ {np_count} pares no-plagiados")

        # 3. PROCESAR PLAGIADOS POR NIVELES
        plag_dir = case_dir / "plagiarized"
        if plag_dir.exists():
            level_totals = {}

            # Procesar cada nivel L1, L2, L3, L4, L5, L6
            for level_name in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
                level_dir = plag_dir / level_name
                if level_dir.exists():
                    level_num = level_name  # Mantener L1, L2, etc.

                    # Buscar directorios numerados y ordenarlos
                    numeric_dirs = sorted([
                        d for d in level_dir.iterdir()
                        if d.is_dir() and d.name.isdigit()
                    ], key=lambda x: int(x.name))

                    version_count = 0
                    for version_dir in numeric_dirs:
                        java_files = list(version_dir.glob("*.java"))
                        if java_files:
                            version_count += 1

                            # Crear nombre de carpeta compacto
                            folder_name = f"{case_id}_orig_{level_num}v{version_count:02d}"
                            pair_folder = output_path / folder_name
                            pair_folder.mkdir(exist_ok=True)

                            # Copiar archivos con nombres estándar
                            orig_dest = pair_folder / "original.java"
                            comp_dest = pair_folder / "compared.java"

                            shutil.copy2(original_file, orig_dest)
                            shutil.copy2(java_files[0], comp_dest)

                            # Registrar metadata
                            pairs_metadata.append({
                                'folder_name': folder_name,
                                'case_id': case_id,
                                'case_orig': case_name,
                                'file1': 'original.java',
                                'file2': 'compared.java',
                                'label': 1,  # Plagiado
                                'plagiarism_level': f'level_{level_name[1:]}',  # level_1, level_2, etc.
                                'source_dataset': 'ir_plag',
                                'comparison_type': f'original_vs_{level_name.lower()}'
                            })
                            total_pairs += 1

                    if version_count > 0:
                        level_totals[level_name] = version_count

            # Mostrar resumen de niveles procesados
            if level_totals:
                levels_summary = ", ".join([f"{k}({v})" for k, v in level_totals.items()])
                print(f"    ✅ Plagiados: {levels_summary}")

    print(f"✅ IR-Plag: {total_pairs} pares creados")
    return pairs_metadata

def process_conplag_data(conplag_path: str, output_path: Path):
    """
    Procesa ConPlag: mueve carpetas de version_2/ y lee labels.csv
    """

    conplag_path = Path(conplag_path)

    print(f"\n🔄 Procesando ConPlag...")
    print(f"📂 Entrada: {conplag_path}")

    # Verificar estructura de ConPlag
    version2_dir = conplag_path / "versions" / "version_2"
    labels_file = conplag_path / "versions" / "labels.csv"

    if not version2_dir.exists():
        print(f"❌ No se encontró {version2_dir}")
        return []

    if not labels_file.exists():
        print(f"❌ No se encontró {labels_file}")
        return []

    # Cargar labels.csv
    print(f"📊 Cargando labels.csv...")
    labels_df = pd.read_csv(labels_file)
    print(f"   {len(labels_df)} pares en labels.csv")

    # Crear índice de labels para búsqueda rápida
    labels_index = {}
    for _, row in labels_df.iterrows():
        key = f"{row['sub1']}_{row['sub2']}"
        labels_index[key] = {
            'problem': row['problem'],
            'verdict': row['verdict']
        }

    # Obtener directorios de pares
    pair_directories = [d for d in version2_dir.iterdir() if d.is_dir()]
    print(f"📁 Encontrados {len(pair_directories)} directorios en version_2/")

    pairs_metadata = []
    moved_pairs = 0
    missing_labels = 0

    for pair_dir in pair_directories:
        dir_name = pair_dir.name

        # Verificar que el directorio tiene el formato esperado
        if '_' not in dir_name:
            print(f"⚠️ Formato inesperado: {dir_name}")
            continue

        try:
            sub1_id, sub2_id = dir_name.split('_', 1)

            # Buscar en labels
            if dir_name in labels_index:
                label_info = labels_index[dir_name]

                # Verificar que los archivos existen
                file1_path = pair_dir / f"{sub1_id}.java"
                file2_path = pair_dir / f"{sub2_id}.java"

                if file1_path.exists() and file2_path.exists():
                    # Crear carpeta en el directorio unificado
                    # Mantener el nombre original de ConPlag
                    dest_folder = output_path / dir_name
                    dest_folder.mkdir(exist_ok=True)

                    # Copiar archivos manteniendo nombres originales
                    shutil.copy2(file1_path, dest_folder / f"{sub1_id}.java")
                    shutil.copy2(file2_path, dest_folder / f"{sub2_id}.java")

                    # Registrar metadata
                    pairs_metadata.append({
                        'folder_name': dir_name,
                        'case_id': f"p{label_info['problem']:02d}",  # p19, p20, etc.
                        'case_orig': f"problem-{label_info['problem']}",
                        'file1': f"{sub1_id}.java",
                        'file2': f"{sub2_id}.java",
                        'label': label_info['verdict'],
                        'plagiarism_level': 'contest_level' if label_info['verdict'] == 1 else 'none',
                        'source_dataset': 'conplag',
                        'comparison_type': 'contest_submission_pair'
                    })
                    moved_pairs += 1
                else:
                    print(f"⚠️ Archivos faltantes en {dir_name}")
            else:
                missing_labels += 1
                if missing_labels <= 5:  # Solo mostrar primeros errores
                    print(f"⚠️ No encontrado en labels: {dir_name}")

        except Exception as e:
            print(f"❌ Error procesando {dir_name}: {e}")

    print(f"✅ ConPlag: {moved_pairs} pares procesados")
    if missing_labels > 0:
        print(f"⚠️ {missing_labels} pares sin labels encontrados")

    return pairs_metadata

def create_unified_csv(ir_plag_metadata: list, conplag_metadata: list, output_path: Path):
    """
    Crea CSV unificado con metadata de ambos datasets
    """

    print(f"\n📝 Creando CSV unificado...")

    # Combinar metadatos
    all_metadata = ir_plag_metadata + conplag_metadata

    # Crear DataFrame
    df = pd.DataFrame(all_metadata)

    # Agregar pair_id único
    df['pair_id'] = [f"pair_{i+1:06d}" for i in range(len(df))]

    # Reordenar columnas para claridad
    columns_order = [
        'pair_id', 'folder_name', 'case_id', 'case_orig',
        'file1', 'file2', 'label', 'plagiarism_level',
        'source_dataset', 'comparison_type'
    ]
    df = df[columns_order]

    # Guardar CSV principal
    csv_file = output_path / "unified_dataset.csv"
    df.to_csv(csv_file, index=False)

    # Crear splits train/test
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df_shuffled))

    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]

    train_df.to_csv(output_path / "train_dataset.csv", index=False)
    test_df.to_csv(output_path / "test_dataset.csv", index=False)

    # Estadísticas generales
    total_pairs = len(df)
    plagiarized = len(df[df['label'] == 1])
    non_plagiarized = len(df[df['label'] == 0])
    ir_plag_count = len(df[df['source_dataset'] == 'ir_plag'])
    conplag_count = len(df[df['source_dataset'] == 'conplag'])

    print(f"✅ CSV unificado creado: {csv_file}")
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total pares: {total_pairs}")
    print(f"   Plagiados: {plagiarized} ({plagiarized/total_pairs*100:.1f}%)")
    print(f"   No plagiados: {non_plagiarized} ({non_plagiarized/total_pairs*100:.1f}%)")
    print(f"   IR-Plag: {ir_plag_count} pares")
    print(f"   ConPlag: {conplag_count} pares")

    # Distribución por nivel de plagio
    print(f"\n📈 DISTRIBUCIÓN POR NIVEL:")
    level_counts = df[df['label'] == 1]['plagiarism_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"   {level}: {count} pares")

    # Estadísticas por dataset
    print(f"\n📂 POR DATASET:")
    for dataset in ['ir_plag', 'conplag']:
        dataset_df = df[df['source_dataset'] == dataset]
        dataset_plag = len(dataset_df[dataset_df['label'] == 1])
        print(f"   {dataset}: {len(dataset_df)} pares ({dataset_plag} plagiados - {dataset_plag/len(dataset_df)*100:.1f}%)")

    return df

def verify_unified_dataset(output_path: Path):
    """
    Verifica que el dataset unificado se creó correctamente
    """

    print(f"\n🔍 Verificando dataset unificado...")

    # Contar carpetas y archivos
    folders = [d for d in output_path.iterdir() if d.is_dir()]
    print(f"📁 Total de carpetas: {len(folders)}")

    # Verificar estructura de carpetas
    ir_plag_folders = [f for f in folders if f.name.startswith('c') and '_orig_' in f.name]
    conplag_folders = [f for f in folders if '_' in f.name and not f.name.startswith('c')]

    print(f"   IR-Plag (formato compacto): {len(ir_plag_folders)}")
    print(f"   ConPlag (formato original): {len(conplag_folders)}")

    # Verificar que cada carpeta tiene 2 archivos Java
    verified_folders = 0
    for folder in folders[:20]:  # Verificar una muestra
        java_files = list(folder.glob("*.java"))
        if len(java_files) == 2:
            verified_folders += 1

    print(f"✅ Carpetas con 2 archivos Java: {verified_folders}/20 verificadas")

    # Mostrar ejemplos de nombres
    print(f"\n📝 Ejemplos de carpetas:")
    if ir_plag_folders:
        print(f"   IR-Plag: {ir_plag_folders[0].name}, {ir_plag_folders[-1].name if len(ir_plag_folders) > 1 else ''}")
    if conplag_folders:
        print(f"   ConPlag: {conplag_folders[0].name}, {conplag_folders[-1].name if len(conplag_folders) > 1 else ''}")

def main():
    # Configurar rutas
    ir_plag_input = "data/IR-Plag-Dataset"
    conplag_input = "data/conplag"
    output_dir = "data/unified_dataset"

    print("🚀 CONVERTIDOR UNIFICADO IR-PLAG + CONPLAG")
    print("=" * 50)
    print(f"📂 IR-Plag entrada: {ir_plag_input}")
    print(f"📂 ConPlag entrada: {conplag_input}")
    print(f"📁 Salida unificada: {output_dir}")
    print()

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Procesar IR-Plag (formato compacto)
    ir_plag_metadata = convert_ir_plag_compact(ir_plag_input, output_path)

    # 2. Procesar ConPlag (mover carpetas version_2)
    conplag_metadata = process_conplag_data(conplag_input, output_path)

    # 3. Crear CSV unificado
    if ir_plag_metadata or conplag_metadata:
        unified_df = create_unified_csv(ir_plag_metadata, conplag_metadata, output_path)

        # 4. Verificar resultado
        verify_unified_dataset(output_path)

        print(f"\n🎉 ¡DATASET UNIFICADO COMPLETADO!")
        print(f"📁 {len(ir_plag_metadata + conplag_metadata)} pares totales")
        print(f"📄 CSV unificado con metadata completa")
        print(f"✅ Ambos datasets en formato consistente")
        print(f"📂 Revisa el resultado en: {output_dir}")
    else:
        print("❌ No se pudieron procesar los datasets")

if __name__ == "__main__":
    main()
