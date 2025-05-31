#!/usr/bin/env python3
"""
Script principal actualizado para construir el dataset unificado
Incorpora los ajustes que funcionaron en la prueba rápida
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import itertools
from typing import List, Tuple, Dict
import random
import json

class UnifiedDatasetBuilder:
    def __init__(self, conplag_path: str, ir_plag_path: str, output_path: str):
        self.conplag_path = Path(conplag_path)
        self.ir_plag_path = Path(ir_plag_path)
        self.output_path = Path(output_path)
        self.cases_info = {}
        self.java_files_index = {}  # Índice para archivos Java de ConPlag

    def create_unified_structure(self):
        """Crea la estructura unificada por casos"""
        self.output_path.mkdir(exist_ok=True)
        print("✓ Estructura base creada")

    def _build_java_files_index(self):
        """Construye un índice de archivos Java para acceso rápido"""
        print("  🔍 Indexando archivos Java de ConPlag...")
        self.java_files_index = {}

        # Buscar en la estructura version2 con directorios de pares
        version2_dir = self.conplag_path / "versions" / "version_2"
        if version2_dir.exists():
            print(f"    📂 Explorando version_2/...")
            pair_dirs = [d for d in version2_dir.iterdir() if d.is_dir()]
            print(f"    📁 Encontrados {len(pair_dirs)} directorios de pares")

            for pair_dir in pair_dirs:
                if '_' in pair_dir.name:
                    try:
                        sub1_id, sub2_id = pair_dir.name.split('_', 1)

                        # Buscar archivos específicos
                        sub1_file = pair_dir / f"{sub1_id}.java"
                        sub2_file = pair_dir / f"{sub2_id}.java"

                        if sub1_file.exists():
                            self.java_files_index[sub1_id] = sub1_file
                        if sub2_file.exists():
                            self.java_files_index[sub2_id] = sub2_file

                    except Exception as e:
                        print(f"      ⚠️ Error procesando {pair_dir.name}: {e}")
        else:
            print(f"    ❌ No se encontró directorio version2/")

        print(f"  ✅ Indexados {len(self.java_files_index)} archivos Java")
        if len(self.java_files_index) > 0:
            # Mostrar algunos ejemplos de IDs indexados
            sample_ids = list(self.java_files_index.keys())[:5]
            print(f"    💡 Ejemplos de IDs indexados: {sample_ids}")

        return len(self.java_files_index)

    def _find_conplag_file(self, submission_id: str) -> Path:
        """Encuentra archivo de submission usando el índice"""
        # Asegurar que el índice existe
        if not hasattr(self, 'java_files_index') or not self.java_files_index:
            self._build_java_files_index()

        # Buscar en el índice
        if submission_id in self.java_files_index:
            return self.java_files_index[submission_id]

        # Si no se encuentra, buscar variaciones
        submission_base = submission_id.replace('.java', '')
        if submission_base in self.java_files_index:
            return self.java_files_index[submission_base]

        return None

    def _group_conplag_by_problem(self, labels_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Agrupa pares de ConPlag por problema"""
        problems = labels_df['problem'].unique()
        groups = {}

        for problem in problems:
            problem_pairs = labels_df[labels_df['problem'] == problem]
            groups[problem] = problem_pairs

        print(f"  📋 Encontrados {len(groups)} problemas diferentes")
        for problem, pairs in list(groups.items())[:5]:  # Mostrar solo primeros 5
            plagiarized = len(pairs[pairs['verdict'] == 1])
            print(f"    Problema {problem}: {len(pairs)} pares ({plagiarized} plagiados)")
        if len(groups) > 5:
            print(f"    ... y {len(groups) - 5} problemas más")

        return groups

    def _verify_file_connections(self, sample_df: pd.DataFrame):
        """Verifica conexión entre labels y archivos en una muestra"""
        print(f"  🔗 Verificando conexión labels -> archivos (muestra de {len(sample_df)})...")

        found_pairs = 0
        missing_files = []

        for _, row in sample_df.iterrows():
            sub1_id = row['sub1']
            sub2_id = row['sub2']

            file1 = self._find_conplag_file(sub1_id)
            file2 = self._find_conplag_file(sub2_id)

            if file1 and file2 and file1.exists() and file2.exists():
                found_pairs += 1
            else:
                if not file1 or not file1.exists():
                    missing_files.append(sub1_id)
                if not file2 or not file2.exists():
                    missing_files.append(sub2_id)

        print(f"    ✅ {found_pairs}/{len(sample_df)} pares con archivos encontrados")

        if found_pairs == 0:
            print("    ❌ PROBLEMA: No se encontraron archivos para ningún par")
            print("    💡 Archivos faltantes de muestra:", missing_files[:5])
        elif found_pairs < len(sample_df):
            print(f"    ⚠️  {len(sample_df) - found_pairs} pares con archivos faltantes")
            if missing_files:
                unique_missing = list(set(missing_files))
                print(f"    💡 IDs no encontrados: {unique_missing[:5]}")

    def process_ir_plag_cases(self):
        """Procesa los casos de IR-Plag manteniendo su estructura"""
        print("Procesando casos de IR-Plag...")

        if not self.ir_plag_path.exists():
            print(f"  ⚠️  Ruta de IR-Plag no existe: {self.ir_plag_path}")
            return

        case_dirs = sorted([d for d in self.ir_plag_path.glob("Case-*") if d.is_dir()])

        if not case_dirs:
            # Buscar patrones alternativos
            alt_patterns = ["case-*", "CASE-*", "task-*", "Task-*"]
            for pattern in alt_patterns:
                case_dirs = sorted([d for d in self.ir_plag_path.glob(pattern) if d.is_dir()])
                if case_dirs:
                    print(f"  📁 Encontrados casos con patrón: {pattern}")
                    break

        if not case_dirs:
            print(f"  ⚠️  No se encontraron casos en {self.ir_plag_path}")
            print(f"  💡 Buscando subdirectorios disponibles...")
            subdirs = [d.name for d in self.ir_plag_path.iterdir() if d.is_dir()]
            print(f"      Directorios encontrados: {subdirs[:10]}")
            return

        print(f"  📁 Encontrados {len(case_dirs)} casos de IR-Plag")
        total_files_processed = 0

        for case_dir in case_dirs:
            case_name = case_dir.name  # Case-1, Case-2, etc.
            output_case_dir = self.output_path / case_name
            output_case_dir.mkdir(exist_ok=True)

            print(f"  📂 Procesando {case_name}...")
            files_copied_this_case = 0

            # 1. CÓDIGO ORIGINAL
            original_dir = output_case_dir / "codigo_original"
            original_dir.mkdir(exist_ok=True)

            source_original = case_dir / "Original"
            if not source_original.exists():
                # Buscar variaciones
                for alt_name in ["original", "ORIGINAL", "src", "source"]:
                    alt_path = case_dir / alt_name
                    if alt_path.exists():
                        source_original = alt_path
                        break

            if source_original.exists():
                java_files = list(source_original.glob("*.java"))
                for java_file in java_files:
                    shutil.copy2(java_file, original_dir / java_file.name)
                    files_copied_this_case += 1
                print(f"    ✅ {len(java_files)} archivos originales")
            else:
                print(f"    ⚠️ No se encontró directorio Original en {case_name}")

            # 2. NO PLAGIO (implementaciones independientes)
            no_plagio_dir = output_case_dir / "no_plagio"
            no_plagio_dir.mkdir(exist_ok=True)

            source_non_plag = case_dir / "non-plagiarized"
            if not source_non_plag.exists():
                # Buscar variaciones
                for alt_name in ["nonplagiarized", "non_plagiarized", "independent", "clean"]:
                    alt_path = case_dir / alt_name
                    if alt_path.exists():
                        source_non_plag = alt_path
                        break

            if source_non_plag.exists():
                version_count = 1
                # Buscar directorios numerados (01, 02, 03, etc.)
                numeric_dirs = sorted([d for d in source_non_plag.iterdir()
                                       if d.is_dir() and d.name.isdigit()])

                for version_dir in numeric_dirs:
                    target_version_dir = no_plagio_dir / f"version_{version_count:02d}"
                    target_version_dir.mkdir(exist_ok=True)

                    java_files = list(version_dir.glob("*.java"))
                    for java_file in java_files:
                        shutil.copy2(java_file, target_version_dir / java_file.name)
                        files_copied_this_case += 1
                    version_count += 1

                print(f"    ✅ {version_count-1} versiones no plagiadas")
            else:
                print(f"    ⚠️ No se encontró directorio non-plagiarized en {case_name}")

            # 3. PLAGIO (6 niveles L1-L6 de Faidhi & Robinson)
            plagio_dir = output_case_dir / "plagio"
            plagio_dir.mkdir(exist_ok=True)

            source_plag = case_dir / "plagiarized"
            if not source_plag.exists():
                # Buscar variaciones
                for alt_name in ["plagiarised", "plag", "modified"]:
                    alt_path = case_dir / alt_name
                    if alt_path.exists():
                        source_plag = alt_path
                        break

            levels_found = 0
            if source_plag.exists():
                # Buscar niveles L1, L2, L3, L4, L5, L6
                level_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']

                for level_name in level_names:
                    level_dir = source_plag / level_name
                    if level_dir.exists():
                        # Convertir L1 -> level_1 para consistencia
                        target_level_name = f"level_{level_name[1:]}"
                        target_level_dir = plagio_dir / target_level_name
                        target_level_dir.mkdir(exist_ok=True)

                        version_count = 1
                        # Buscar directorios numerados dentro del nivel
                        numeric_dirs = sorted([d for d in level_dir.iterdir()
                                               if d.is_dir() and d.name.isdigit()])

                        for version_dir in numeric_dirs:
                            target_version_dir = target_level_dir / f"version_{version_count:02d}"
                            target_version_dir.mkdir(exist_ok=True)

                            java_files = list(version_dir.glob("*.java"))
                            for java_file in java_files:
                                shutil.copy2(java_file, target_version_dir / java_file.name)
                                files_copied_this_case += 1
                            version_count += 1

                        levels_found += 1
                        print(f"    ✅ {level_name} -> level_{level_name[1:]} ({version_count-1} versiones)")

                print(f"    ✅ {levels_found} niveles de plagio procesados")
            else:
                print(f"    ⚠️ No se encontró directorio plagiarized en {case_name}")

            total_files_processed += files_copied_this_case

            # Registrar información del caso
            self.cases_info[case_name] = {
                'source': 'ir_plag',
                'original_files': len(list(original_dir.glob("*.java"))),
                'no_plagio_versions': len(list(no_plagio_dir.glob("version_*"))),
                'plagio_levels': len(list(plagio_dir.glob("level_*"))),
                'files_copied': files_copied_this_case,
                'description': f'Caso académico de IR-Plag con {levels_found} niveles de transformación'
            }

            print(f"    📄 {files_copied_this_case} archivos copiados")

        print(f"✓ Procesados {len(case_dirs)} casos de IR-Plag")
        print(f"📄 Total archivos procesados: {total_files_processed}")

    def process_conplag_cases(self):
        """Convierte ConPlag en casos organizados"""
        print("Procesando ConPlag como casos de contest...")

        # Leer etiquetas de ConPlag
        labels_file = self.conplag_path / "versions" / "labels.csv"
        if not labels_file.exists():
            print("  ⚠️  No se encontró labels.csv en ConPlag")
            return

        labels_df = pd.read_csv(labels_file)

        # Mostrar estructura del archivo para debug
        print(f"  📋 Columnas en labels.csv: {list(labels_df.columns)}")
        print(f"  📊 Total de pares: {len(labels_df)}")

        # Verificar si tenemos las columnas esperadas
        expected_cols = ['sub1', 'sub2', 'problem', 'verdict']
        if not all(col in labels_df.columns for col in expected_cols):
            print(f"  ⚠️  Columnas esperadas: {expected_cols}")
            print(f"  ⚠️  Columnas encontradas: {list(labels_df.columns)}")
            return

        # Construir índice de archivos Java
        total_java_files = self._build_java_files_index()
        if total_java_files == 0:
            print("  ❌ No se encontraron archivos Java en ConPlag")
            return

        # Agrupar pares por problema
        pairs_by_problem = self._group_conplag_by_problem(labels_df)

        # Verificar conexión labels -> archivos en una muestra
        self._verify_file_connections(labels_df.head(10))

        case_num = 1
        total_files_processed = 0

        for problem_id, group_pairs in pairs_by_problem.items():
            case_name = f"Contest-{case_num:02d}"
            output_case_dir = self.output_path / case_name
            output_case_dir.mkdir(exist_ok=True)

            print(f"  📁 Creando {case_name} (Problema {problem_id}) con {len(group_pairs)} pares...")

            # Directorios
            original_dir = output_case_dir / "codigo_original"
            no_plagio_dir = output_case_dir / "no_plagio"
            plagio_dir = output_case_dir / "plagio"

            original_dir.mkdir(exist_ok=True)
            no_plagio_dir.mkdir(exist_ok=True)
            plagio_dir.mkdir(exist_ok=True)
            (plagio_dir / "contest_level").mkdir(exist_ok=True)

            # Procesar pares del grupo
            file_counter = 1
            plagio_version = 1
            no_plagio_version = 1
            files_copied_this_case = 0

            for _, pair in group_pairs.iterrows():
                # Usar las columnas correctas de ConPlag
                file1_path = self._find_conplag_file(pair['sub1'])
                file2_path = self._find_conplag_file(pair['sub2'])

                if file1_path and file2_path and file1_path.exists() and file2_path.exists():
                    try:
                        if pair['verdict'] == 1:  # Plagiado
                            # Uno va a original, otro a plagio
                            shutil.copy2(file1_path, original_dir / f"original_{file_counter}.java")

                            version_dir = plagio_dir / "contest_level" / f"version_{plagio_version:02d}"
                            version_dir.mkdir(exist_ok=True)
                            shutil.copy2(file2_path, version_dir / f"plagiarized_{file_counter}.java")
                            plagio_version += 1
                            files_copied_this_case += 2

                        else:  # No plagiado
                            # Ambos van a no_plagio (implementaciones independientes)
                            version_dir1 = no_plagio_dir / f"version_{no_plagio_version:02d}"
                            version_dir1.mkdir(exist_ok=True)
                            shutil.copy2(file1_path, version_dir1 / f"independent_{file_counter}a.java")
                            no_plagio_version += 1

                            version_dir2 = no_plagio_dir / f"version_{no_plagio_version:02d}"
                            version_dir2.mkdir(exist_ok=True)
                            shutil.copy2(file2_path, version_dir2 / f"independent_{file_counter}b.java")
                            no_plagio_version += 1
                            files_copied_this_case += 2

                        file_counter += 1

                    except Exception as e:
                        print(f"    ⚠️ Error copiando {pair['sub1']}-{pair['sub2']}: {e}")
                else:
                    # Contar archivos no encontrados para debug
                    missing = []
                    if not file1_path or not file1_path.exists():
                        missing.append(pair['sub1'])
                    if not file2_path or not file2_path.exists():
                        missing.append(pair['sub2'])
                    if len(missing) > 0 and file_counter <= 5:  # Solo mostrar primeros errores
                        print(f"    ⚠️ Archivos no encontrados: {missing}")

            total_files_processed += files_copied_this_case

            # Registrar información del caso
            self.cases_info[case_name] = {
                'source': 'conplag',
                'problem_id': problem_id,
                'original_files': len(list(original_dir.glob("*.java"))),
                'no_plagio_versions': len(list(no_plagio_dir.glob("version_*"))),
                'plagio_levels': 1,  # Solo nivel de contest
                'files_copied': files_copied_this_case,
                'description': f'Contest submissions - Problema {problem_id} - {len(group_pairs)} pares'
            }

            print(f"    ✅ {files_copied_this_case} archivos copiados")
            case_num += 1

        print(f"✓ Procesados {case_num-1} casos de contest desde ConPlag")
        print(f"📄 Total archivos procesados: {total_files_processed}")

    def generate_pairs_metadata(self):
        """Genera metadata de todos los pares posibles"""
        print("Generando metadata de pares...")

        all_pairs = []
        pair_id = 1

        for case_name in sorted(self.cases_info.keys()):
            case_dir = self.output_path / case_name

            print(f"  Procesando pares de {case_name}...")

            # Obtener archivos de cada categoría
            original_files = list((case_dir / "codigo_original").glob("*.java"))
            no_plagio_files = []
            plagio_files = []

            # Archivos no plagiados
            no_plagio_dir = case_dir / "no_plagio"
            for version_dir in no_plagio_dir.glob("version_*"):
                no_plagio_files.extend(list(version_dir.glob("*.java")))

            # Archivos plagiados
            plagio_dir = case_dir / "plagio"
            for level_dir in plagio_dir.iterdir():
                if level_dir.is_dir():
                    level_name = level_dir.name
                    for version_dir in level_dir.glob("version_*"):
                        for java_file in version_dir.glob("*.java"):
                            plagio_files.append((java_file, level_name))

            # Generar pares NO PLAGIADOS
            # Original vs No-plagiado
            for orig_file in original_files:
                for no_plag_file in no_plagio_files:
                    all_pairs.append({
                        'pair_id': f"pair_{pair_id:06d}",
                        'case': case_name,
                        'file1': str(orig_file.relative_to(self.output_path)),
                        'file2': str(no_plag_file.relative_to(self.output_path)),
                        'label': 0,
                        'plagiarism_level': 'none',
                        'source_dataset': self.cases_info[case_name]['source'],
                        'comparison_type': 'original_vs_independent'
                    })
                    pair_id += 1

            # No-plagiado vs No-plagiado
            for file1, file2 in itertools.combinations(no_plagio_files, 2):
                all_pairs.append({
                    'pair_id': f"pair_{pair_id:06d}",
                    'case': case_name,
                    'file1': str(file1.relative_to(self.output_path)),
                    'file2': str(file2.relative_to(self.output_path)),
                    'label': 0,
                    'plagiarism_level': 'none',
                    'source_dataset': self.cases_info[case_name]['source'],
                    'comparison_type': 'independent_vs_independent'
                })
                pair_id += 1

            # Generar pares PLAGIADOS
            # Original vs Plagiado
            for orig_file in original_files:
                for plag_file, level in plagio_files:
                    all_pairs.append({
                        'pair_id': f"pair_{pair_id:06d}",
                        'case': case_name,
                        'file1': str(orig_file.relative_to(self.output_path)),
                        'file2': str(plag_file.relative_to(self.output_path)),
                        'label': 1,
                        'plagiarism_level': level,
                        'source_dataset': self.cases_info[case_name]['source'],
                        'comparison_type': 'original_vs_plagiarized'
                    })
                    pair_id += 1

            # Plagiado vs Plagiado (mismo nivel)
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
                        'file1': str(file1.relative_to(self.output_path)),
                        'file2': str(file2.relative_to(self.output_path)),
                        'label': 1,
                        'plagiarism_level': level,
                        'source_dataset': self.cases_info[case_name]['source'],
                        'comparison_type': 'plagiarized_vs_plagiarized'
                    })
                    pair_id += 1

        # Crear DataFrame y guardar
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df.to_csv(self.output_path / "all_pairs.csv", index=False)

        # Crear split train/test
        pairs_shuffled = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(0.8 * len(pairs_shuffled))

        train_df = pairs_shuffled[:train_size]
        test_df = pairs_shuffled[train_size:]

        train_df.to_csv(self.output_path / "train_pairs.csv", index=False)
        test_df.to_csv(self.output_path / "test_pairs.csv", index=False)

        print(f"✓ Generados {len(all_pairs)} pares totales")
        print(f"  - Train: {len(train_df)} pares")
        print(f"  - Test: {len(test_df)} pares")

        return pairs_df

    def create_summary_report(self, pairs_df: pd.DataFrame):
        """Crea reporte resumen del dataset unificado"""
        # Estadísticas generales
        stats = {
            'total_cases': len(self.cases_info),
            'total_pairs': len(pairs_df),
            'plagiarized_pairs': len(pairs_df[pairs_df['label'] == 1]),
            'non_plagiarized_pairs': len(pairs_df[pairs_df['label'] == 0]),
            'ir_plag_cases': len([c for c in self.cases_info if self.cases_info[c]['source'] == 'ir_plag']),
            'conplag_cases': len([c for c in self.cases_info if self.cases_info[c]['source'] == 'conplag']),
            'total_files_copied': sum(info.get('files_copied', 0) for info in self.cases_info.values())
        }

        # Guardar información de casos (convertir tipos numpy a nativos)
        cases_info_serializable = {}
        for case, info in self.cases_info.items():
            cases_info_serializable[case] = {}
            for key, value in info.items():
                # Convertir tipos numpy/pandas a tipos nativos de Python
                if hasattr(value, 'item'):
                    cases_info_serializable[case][key] = value.item()
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    cases_info_serializable[case][key] = value
                else:
                    cases_info_serializable[case][key] = str(value)

        with open(self.output_path / "cases_info.json", 'w') as f:
            json.dump(cases_info_serializable, f, indent=2)

        # Crear reporte
        with open(self.output_path / "dataset_summary.txt", 'w') as f:
            f.write("DATASET UNIFICADO - CONPLAG + IR-PLAG\n")
            f.write("=" * 50 + "\n\n")

            f.write("ESTRUCTURA:\n")
            f.write("CasoN/\n")
            f.write("├── codigo_original/    # Implementaciones base\n")
            f.write("├── no_plagio/         # Implementaciones independientes\n")
            f.write("└── plagio/            # Implementaciones plagiadas por nivel\n\n")

            f.write("ESTADÍSTICAS GENERALES:\n")
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

            f.write(f"\nBalance: {stats['plagiarized_pairs']/stats['total_pairs']:.1%} plagiados\n\n")

            f.write("CASOS INCLUIDOS:\n")
            for case_name, info in sorted(self.cases_info.items()):
                f.write(f"  {case_name}: {info['description']}\n")
                f.write(f"    - Originales: {info['original_files']}\n")
                f.write(f"    - No plagio: {info['no_plagio_versions']} versiones\n")
                f.write(f"    - Plagio: {info['plagio_levels']} niveles\n")
                f.write(f"    - Archivos copiados: {info.get('files_copied', 0)}\n")

            # Estadísticas por nivel de plagio
            f.write("\nDISTRIBUCIÓN POR NIVEL DE PLAGIO:\n")
            level_counts = pairs_df[pairs_df['label'] == 1]['plagiarism_level'].value_counts()
            for level, count in level_counts.items():
                f.write(f"  {level}: {count} pares\n")

        print(f"✓ Reporte guardado en {self.output_path / 'dataset_summary.txt'}")

        return stats

    def build_unified_dataset(self):
        """Proceso principal de construcción"""
        print("🚀 Iniciando construcción del dataset unificado...")
        print("📁 Estructura: CasoN/codigo_original|no_plagio|plagio/\n")

        # 1. Crear estructura
        self.create_unified_structure()

        # 2. Procesar IR-Plag (mantener casos)
        self.process_ir_plag_cases()

        # 3. Procesar ConPlag (convertir a casos)
        self.process_conplag_cases()

        # 4. Generar metadata de pares
        pairs_df = self.generate_pairs_metadata()

        # 5. Crear reporte
        stats = self.create_summary_report(pairs_df)

        print("\n🎉 DATASET UNIFICADO COMPLETADO!")
        print(f"📊 {stats['total_cases']} casos, {stats['total_pairs']} pares")
        print(f"📄 {stats['total_files_copied']} archivos copiados")
        print(f"📁 Ubicación: {self.output_path}")

        return stats

# Uso
if __name__ == "__main__":
    # ⚠️ AJUSTAR ESTAS RUTAS SEGÚN TU CONFIGURACIÓN ⚠️
    conplag_path = "data/conplag"
    ir_plag_path = "data/IR-Plag-Dataset"
    output_path = "data/elbueno"

    print("🔧 Construyendo dataset unificado con configuración actualizada...")
    print(f"📂 ConPlag: {conplag_path}")
    print(f"📂 IR-Plag: {ir_plag_path}")
    print(f"📂 Salida: {output_path}")
    print()

    # Construir dataset
    builder = UnifiedDatasetBuilder(conplag_path, ir_plag_path, output_path)
    stats = builder.build_unified_dataset()
