#!/usr/bin/env python3
"""
Convierte IR-Plag a formato compacto y consistente
Formato: c01_orig_np01/, c01_orig_L1v01/, etc.
Cada carpeta contiene exactamente 2 archivos Java
"""

import shutil
from pathlib import Path
import pandas as pd

def convert_ir_plag_compact(ir_plag_path: str, output_path: str):
    """
    Convierte IR-Plag a formato compacto con nombres claros

    Formato de salida:
    - c01_orig_np01/     # caso 1, original vs no-plagio versión 1
    - c01_orig_L1v01/    # caso 1, original vs nivel 1 versión 1
    - c02_orig_L6v05/    # caso 2, original vs nivel 6 versión 5
    """

    ir_plag_path = Path(ir_plag_path)
    output_path = Path(output_path)

    # Crear directorio de salida
    output_path.mkdir(parents=True, exist_ok=True)

    print("🔄 Convirtiendo IR-Plag a formato compacto...")
    print(f"📂 Entrada: {ir_plag_path}")
    print(f"📁 Salida: {output_path}")

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

        print(f"\n📂 Procesando {case_name} → {case_id}")

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
        print(f"    📄 Original: {original_file.name}")

        # 2. PROCESAR NO-PLAGIADOS
        non_plag_dir = case_dir / "non-plagiarized"
        if non_plag_dir.exists():
            print(f"    🔍 Procesando no-plagiados...")

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
                        'case_num': case_number,
                        'case_orig': case_name,
                        'file1': 'original.java',
                        'file2': 'compared.java',
                        'label': 0,  # No plagiado
                        'plagiarism_level': 'none',
                        'comparison_type': 'original_vs_nonplagiarized',
                        'level_detail': f'np{np_count:02d}'
                    })
                    total_pairs += 1

            print(f"      ✅ {np_count} pares no-plagiados creados")

        # 3. PROCESAR PLAGIADOS POR NIVELES
        plag_dir = case_dir / "plagiarized"
        if plag_dir.exists():
            print(f"    🔍 Procesando plagiados...")

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
                                'case_num': case_number,
                                'case_orig': case_name,
                                'file1': 'original.java',
                                'file2': 'compared.java',
                                'label': 1,  # Plagiado
                                'plagiarism_level': f'level_{level_name[1:]}',  # level_1, level_2, etc.
                                'comparison_type': f'original_vs_{level_name.lower()}',
                                'level_detail': f'{level_num}v{version_count:02d}'
                            })
                            total_pairs += 1

                    if version_count > 0:
                        level_totals[level_name] = version_count
                        print(f"      ✅ {level_name}: {version_count} pares plagiados")

    print(f"\n📊 CONVERSIÓN COMPLETADA:")
    print(f"✅ {total_pairs} pares creados en formato compacto")
    print(f"📁 Ubicación: {output_path}")

    return pairs_metadata

def create_ir_plag_csv(pairs_metadata: list, output_path: Path):
    """
    Crea CSV con metadata de todos los pares de IR-Plag
    """

    print("\n📝 Creando CSV de metadata...")

    # Crear DataFrame
    df = pd.DataFrame(pairs_metadata)

    # Agregar información adicional
    df['pair_id'] = [f"ir_{i+1:04d}" for i in range(len(df))]
    df['source_dataset'] = 'ir_plag'

    # Reordenar columnas
    columns_order = [
        'pair_id', 'folder_name', 'case_num', 'case_orig',
        'file1', 'file2', 'label', 'plagiarism_level',
        'comparison_type', 'level_detail', 'source_dataset'
    ]
    df = df[columns_order]

    # Guardar CSV
    csv_file = output_path / "ir_plag_pairs.csv"
    df.to_csv(csv_file, index=False)

    # Estadísticas
    total_pairs = len(df)
    plagiarized = len(df[df['label'] == 1])
    non_plagiarized = len(df[df['label'] == 0])

    print(f"✅ CSV creado: {csv_file}")
    print(f"📊 Estadísticas:")
    print(f"   Total pares: {total_pairs}")
    print(f"   Plagiados: {plagiarized}")
    print(f"   No plagiados: {non_plagiarized}")
    print(f"   Balance: {plagiarized/total_pairs*100:.1f}% plagiados")

    # Mostrar distribución por nivel
    print(f"\n📈 Distribución por nivel:")
    level_counts = df[df['label'] == 1]['plagiarism_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"   {level}: {count} pares")

    # Mostrar casos procesados
    cases_processed = df['case_num'].nunique()
    print(f"\n📁 Casos procesados: {cases_processed}")
    for case_num in sorted(df['case_num'].unique()):
        case_pairs = len(df[df['case_num'] == case_num])
        case_plag = len(df[(df['case_num'] == case_num) & (df['label'] == 1)])
        print(f"   Caso {case_num}: {case_pairs} pares ({case_plag} plagiados)")

    return df

def verify_conversion(output_path: Path):
    """
    Verifica que la conversión se hizo correctamente
    """

    print(f"\n🔍 Verificando conversión...")

    # Contar carpetas creadas
    pair_folders = [d for d in output_path.iterdir() if d.is_dir()]
    print(f"📁 Carpetas creadas: {len(pair_folders)}")

    # Verificar estructura de algunas carpetas
    verified_folders = 0
    errors = []

    for folder in pair_folders[:10]:  # Verificar primeras 10
        java_files = list(folder.glob("*.java"))

        if len(java_files) == 2:
            expected_files = ['original.java', 'compared.java']
            actual_files = [f.name for f in java_files]

            if all(ef in actual_files for ef in expected_files):
                verified_folders += 1
            else:
                errors.append(f"{folder.name}: archivos {actual_files} (esperados: {expected_files})")
        else:
            errors.append(f"{folder.name}: {len(java_files)} archivos (esperados: 2)")

    print(f"✅ Carpetas verificadas correctamente: {verified_folders}/10")

    if errors:
        print(f"⚠️ Errores encontrados:")
        for error in errors[:5]:
            print(f"   - {error}")

    # Mostrar ejemplos de nombres de carpetas
    print(f"\n📝 Ejemplos de nombres de carpetas:")
    sample_folders = sorted([d.name for d in pair_folders])

    # Mostrar algunos ejemplos organizados
    np_examples = [f for f in sample_folders if '_np' in f][:3]
    l1_examples = [f for f in sample_folders if '_L1v' in f][:3]
    l6_examples = [f for f in sample_folders if '_L6v' in f][:3]

    if np_examples:
        print(f"   No-plagio: {', '.join(np_examples)}")
    if l1_examples:
        print(f"   Nivel 1: {', '.join(l1_examples)}")
    if l6_examples:
        print(f"   Nivel 6: {', '.join(l6_examples)}")

def main():
    # Configurar rutas
    ir_plag_input = "data/IR-Plag-Dataset"
    output_dir = "data/ir_plag_compact"

    print("🚀 CONVERTIDOR IR-PLAG A FORMATO COMPACTO")
    print("=" * 45)
    print(f"📂 Entrada: {ir_plag_input}")
    print(f"📁 Salida: {output_dir}")
    print()
    print("🏷️ Formato de nombres:")
    print("   c01_orig_np01/     # caso 1, original vs no-plagio 1")
    print("   c01_orig_L1v01/    # caso 1, original vs nivel 1 versión 1")
    print("   c02_orig_L6v05/    # caso 2, original vs nivel 6 versión 5")
    print()

    # 1. Convertir IR-Plag
    pairs_metadata = convert_ir_plag_compact(ir_plag_input, output_dir)

    if not pairs_metadata:
        print("❌ No se pudieron procesar los datos de IR-Plag")
        return

    # 2. Crear CSV
    df = create_ir_plag_csv(pairs_metadata, Path(output_dir))

    # 3. Verificar conversión
    verify_conversion(Path(output_dir))

    print(f"\n🎉 ¡CONVERSIÓN COMPLETADA!")
    print(f"📁 {len(pairs_metadata)} pares creados en formato compacto")
    print(f"📄 CSV generado con metadata completa")
    print(f"✅ Cada carpeta contiene exactamente 2 archivos Java")
    print(f"📂 Revisa el resultado en: {output_dir}")

if __name__ == "__main__":
    main()
