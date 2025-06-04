#!/usr/bin/env python3
"""
Dvides a dataset into train, validation, and test sets
"""

import pandas as pd
import shutil
from pathlib import Path
import random
import argparse
from typing import Tuple, List

def load_dataset_info(dataset_path: Path) -> pd.DataFrame:
    """
    Loads the dataset information from a CSV file
    :param dataset_path: Path to the dataset directory
    :return: DataFrame with dataset information
    """

    csv_file = dataset_path / "unified_dataset.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"No se encontró {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Cargado dataset con {len(df)} pares")

    return df

def verify_folders_exist(dataset_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifies that the folders in the dataset exist and contain at least 2 Java files
    :param dataset_path: Path to the dataset directory
    :param df: DataFrame with dataset information
    :return: DataFrame with valid rows (folders that exist and have at least 2 Java files)
    """

    print("Verificando existencia de carpetas")

    valid_rows = []
    missing_folders = []

    for _, row in df.iterrows():
        folder_path = dataset_path / row['folder_name']

        if folder_path.exists() and folder_path.is_dir():
            # Verify that there are at least 2 Java files
            java_files = list(folder_path.glob("*.java"))
            if len(java_files) >= 2:
                valid_rows.append(row)
            else:
                missing_folders.append(f"{row['folder_name']} (archivos faltantes)")
        else:
            missing_folders.append(f"{row['folder_name']} (carpeta no existe)")

    if missing_folders:
        print(f"{len(missing_folders)} carpetas con problemas:")
        for folder in missing_folders[:5]:
            print(f"   - {folder}")
        if len(missing_folders) > 5:
            print(f"   ... y {len(missing_folders) - 5} más")

    valid_df = pd.DataFrame(valid_rows)
    print(f"{len(valid_df)}/{len(df)} carpetas validas")

    return valid_df

def create_stratified_splits(df: pd.DataFrame, train_ratio: float = 0.7,
                             val_ratio: float = 0.1, test_ratio: float = 0.2,
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates stratified splits of the dataset
    :param df: DataFrame with dataset information
    :param train_ratio: Proportion of data for training set
    :param val_ratio: Proportion of data for validation set
    :param test_ratio: Proportion of data for test set
    :param random_seed: Random seed for reproducibility
    :return: Tuple of DataFrames for train, validation, and test sets
    """

    print(f"\nCreando splits")
    print(f"   Train: {train_ratio*100:.0f}%, Validation: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")

    # Verify that the proportions sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Las proporciones no suman 1.0")

    random.seed(random_seed)

    splits = {'train': [], 'validation': [], 'test': []}

    for source in df['source_dataset'].unique():
        for label in df['label'].unique():

            subset = df[(df['source_dataset'] == source) & (df['label'] == label)]

            if len(subset) == 0:
                continue

            subset_shuffled = subset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            total_size = len(subset_shuffled)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)

            train_subset = subset_shuffled[:train_size]
            val_subset = subset_shuffled[train_size:train_size + val_size]
            test_subset = subset_shuffled[train_size + val_size:]

            splits['train'].append(train_subset)
            splits['validation'].append(val_subset)
            splits['test'].append(test_subset)

            print(f"   {source} label={label}: {len(train_subset)} train, {len(val_subset)} val, {len(test_subset)} test")

    train_df = pd.concat(splits['train'], ignore_index=True) if splits['train'] else pd.DataFrame()
    val_df = pd.concat(splits['validation'], ignore_index=True) if splits['validation'] else pd.DataFrame()
    test_df = pd.concat(splits['test'], ignore_index=True) if splits['test'] else pd.DataFrame()

    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)

    print(f"\nSplits creados:")
    print(f"   Train: {len(train_df)} pares")
    print(f"   Validation: {len(val_df)} pares")
    print(f"   Test: {len(test_df)} pares")

    return train_df, val_df, test_df

def copy_folders_for_split(dataset_path: Path, split_df: pd.DataFrame,
                           split_name: str, output_path: Path) -> Path:
    """
    Copy folders for a specific split
    :param dataset_path: Path to the original dataset directory
    :param split_df: DataFrame with the split information
    :param split_name: Name of the split (train, validation, test)
    :param output_path: Path to the output directory where the split will be saved
    :return: Path to the directory where the split was saved
    """

    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopiando carpetas para {split_name}")

    copied_folders = 0

    for _, row in split_df.iterrows():
        source_folder = dataset_path / row['folder_name']
        dest_folder = split_dir / row['folder_name']

        if source_folder.exists():
            try:
                if dest_folder.exists():
                    shutil.rmtree(dest_folder)
                shutil.copytree(source_folder, dest_folder)
                copied_folders += 1
            except Exception as e:
                print(f"Error copiando {row['folder_name']}: {e}")

    print(f"{copied_folders}/{len(split_df)} carpetas copiadas para {split_name}")

    return split_dir

def save_split_csv(split_df: pd.DataFrame, split_name: str, output_path: Path):
    """
    Saves the split DataFrame to a CSV file
    :param split_df: DataFrame with the split information
    :param split_name: Name of the split (train, validation, test)
    :param output_path: Path to the output directory where the CSV will be saved
    :return: None
    """

    csv_file = output_path / f"{split_name}.csv"
    split_df.to_csv(csv_file, index=False)

    print(f"CSV guardado: {csv_file}")

def analyze_split_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame,
                             test_df: pd.DataFrame):
    """
    Analyzes and prints statistics for each split
    :param train_df: DataFrame for the training set
    :param val_df: DataFrame for the validation set
    :param test_df: DataFrame for the test set
    :return: None
    """

    print(f"\nANALISIS DE ESTADÍSTICAS POR SPLIT:")
    print("=" * 50)

    splits_data = {
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }

    for split_name, split_df in splits_data.items():
        if len(split_df) == 0:
            continue

        total = len(split_df)
        plagiarized = len(split_df[split_df['label'] == 1])
        non_plagiarized = total - plagiarized

        ir_plag_count = len(split_df[split_df['source_dataset'] == 'ir_plag'])
        conplag_count = len(split_df[split_df['source_dataset'] == 'conplag'])

        print(f"\n{split_name.upper()}:")
        print(f"   Total pares: {total}")
        print(f"   Plagiados: {plagiarized} ({plagiarized/total*100:.1f}%)")
        print(f"   No plagiados: {non_plagiarized} ({non_plagiarized/total*100:.1f}%)")
        print(f"   IR-Plag: {ir_plag_count} ({ir_plag_count/total*100:.1f}%)")
        print(f"   ConPlag: {conplag_count} ({conplag_count/total*100:.1f}%)")

        # Distribution of plagiarism levels
        if plagiarized > 0:
            level_counts = split_df[split_df['label'] == 1]['plagiarism_level'].value_counts()
            levels_summary = ", ".join([f"{k}({v})" for k, v in level_counts.items()])
            print(f"   Niveles: {levels_summary}")

def create_summary_file(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, output_path: Path):
    """
    Creates a summary file with statistics of the dataset splits
    :param train_df: DataFrame for the training set
    :param val_df: DataFrame for the validation set
    :param test_df: DataFrame for the test set
    :param output_path: Path to the output directory where the summary will be saved
    :return: None
    """

    summary_file = output_path / "splits_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("RESUMEN DE DIVISION DEL DATASET\n")
        f.write("=" * 40 + "\n\n")

        total_original = len(train_df) + len(val_df) + len(test_df)

        f.write("CONFIGURACION:\n")
        f.write(f"  Train: {len(train_df)} pares ({len(train_df)/total_original*100:.1f}%)\n")
        f.write(f"  Validation: {len(val_df)} pares ({len(val_df)/total_original*100:.1f}%)\n")
        f.write(f"  Test: {len(test_df)} pares ({len(test_df)/total_original*100:.1f}%)\n")
        f.write(f"  Total: {total_original} pares\n\n")

        splits_data = {'Train': train_df, 'Validation': val_df, 'Test': test_df}

        for split_name, split_df in splits_data.items():
            if len(split_df) == 0:
                continue

            f.write(f"{split_name.upper()}:\n")
            f.write(f"  Total: {len(split_df)} pares\n")

            plagiarized = len(split_df[split_df['label'] == 1])
            f.write(f"  Plagiados: {plagiarized} ({plagiarized/len(split_df)*100:.1f}%)\n")

            ir_plag = len(split_df[split_df['source_dataset'] == 'ir_plag'])
            conplag = len(split_df[split_df['source_dataset'] == 'conplag'])
            f.write(f"  IR-Plag: {ir_plag}, ConPlag: {conplag}\n")

            unique_cases = split_df['case_id'].nunique()
            f.write(f"  Casos únicos: {unique_cases}\n\n")

    print(f"resumen guardado en: {summary_file}")

def split_dataset(dataset_path: str, output_path: str = None,
                  train_ratio: float = 0.7, val_ratio: float = 0.1,
                  test_ratio: float = 0.2, random_seed: int = 42):
    """
    Divides a dataset into train, validation, and test sets.
    :param dataset_path: Path to the dataset directory
    :param output_path: Path to the output directory where the splits will be saved
    :param train_ratio: Proportion of data for training set (default: 0.7)
    :param val_ratio: Proportion of data for validation set (default: 0.1)
    :param test_ratio: Proportion of data for test set (default: 0.2)
    :param random_seed: Random seed for reproducibility (default: 42)
    :return: None
    """

    dataset_path = Path(dataset_path)

    if output_path is None:
        output_path = Path("data/splits")
    else:
        output_path = Path(output_path)

    print("DIVISION DE DATASET EN TRAIN/VALIDATION/TEST")
    print("=" * 55)
    print(f"Dataset origen: {dataset_path}")
    print(f"Salida: {output_path}")
    print(f"Seed aleatoria: {random_seed}")

    df = load_dataset_info(dataset_path)

    valid_df = verify_folders_exist(dataset_path, df)

    if len(valid_df) == 0:
        print("No se encontraron carpetas validas")
        return

    train_df, val_df, test_df = create_stratified_splits(
        valid_df, train_ratio, val_ratio, test_ratio, random_seed
    )

    output_path.mkdir(parents=True, exist_ok=True)

    copy_folders_for_split(dataset_path, train_df, "train", output_path)
    copy_folders_for_split(dataset_path, val_df, "validation", output_path)
    copy_folders_for_split(dataset_path, test_df, "test", output_path)

    save_split_csv(train_df, "train", output_path)
    save_split_csv(val_df, "validation", output_path)
    save_split_csv(test_df, "test", output_path)

    analyze_split_statistics(train_df, val_df, test_df)

    create_summary_file(train_df, val_df, test_df, output_path)

    print(f"\n Division completada")
    print(f"resultados en: {output_path}")
    print(f"3 carpetas creadas: train/, validation/, test/")
    print(f"3 CSVs creados + resumen")

def main():
    """
    Main function to parse arguments and execute the dataset splitting
    :return: None
    """
    parser = argparse.ArgumentParser(description='Divide dataset en train/validation/test')
    parser.add_argument('dataset_path', help='Ruta al dataset unificado')
    parser.add_argument('--output', default="data/splits", help='Directorio de salida (default: data/splits)')
    parser.add_argument('--train', type=float, default=0.7, help='Proporción para train (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.1, help='Proporción para validation (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.2, help='Proporción para test (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42)')

    args = parser.parse_args()

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error las proporciones suman {total_ratio}, deben sumar 1.0")
        return

    split_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()
