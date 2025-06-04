from sklearn.metrics import classification_report

from astcc import *

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Generates and saves a confusion matrix
    :param y_true: Etiquetas verdaderas
    :param y_pred: Etiquetas predichas
    :param title: Título de la matriz
    :param filename: Nombre del archivo donde se guardará la matriz
    :return: None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Plagio', 'Plagio'],
                yticklabels=['No Plagio', 'Plagio'])
    plt.title(title)
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusión guardada en: {filename}")

def compare_plagiarism_detectors(base_path="data/splits/train/",
                                 csv_path="data/splits/train.csv",
                                 output_csv="comparison_results.csv"):
    """
    Compare TF-IDF and AST-CC plagiarism detection methods and output results to CSV
    :param base_path: Base path where the dataset is located
    :param csv_path: Path to the CSV file containing dataset information
    :param output_csv: Path to save the comparison results
    :return: DataFrame with comparison results
    """
    # Initialize detectors
    tfidf_detector = TFIDFPlagiarismDetector()
    astcc_detector = ASTCCDetector()

    # Load dataset
    df = pd.read_csv(csv_path)
    if 'source_dataset' in df.columns:
        df = df[df['source_dataset'].isin(['ir_plag', 'conplag'])]

    print(f"Analyzing {len(df)} file pairs with both methods")
    results = []

    for idx, row in df.iterrows():

        # Extract file information
        label = row['label']
        dataset = row.get('source_dataset', 'unknown')
        file1 = row.get('file1')
        file2 = row.get('file2')
        folder_name = row.get('folder_name')

        # Build file paths based on dataset structure
        if dataset == 'ir_plag' and folder_name:
            folder_path = os.path.join(base_path, folder_name)
            path1 = os.path.join(folder_path, 'original.java')
            path2 = os.path.join(folder_path, 'compared.java')
            folder = folder_name
        elif dataset == 'conplag' and file1 and file2:
            file1_base = os.path.splitext(file1)[0]
            file2_base = os.path.splitext(file2)[0]

            folder1 = os.path.join(base_path, f"{file1_base}_{file2_base}")
            folder2 = os.path.join(base_path, f"{file2_base}_{file1_base}")

            if os.path.isdir(folder1):
                folder_path = folder1
                folder = f"{file1_base}_{file2_base}"
            elif os.path.isdir(folder2):
                folder_path = folder2
                folder = f"{file2_base}_{file1_base}"
            else:
                continue

            path1 = os.path.join(folder_path, file1)
            path2 = os.path.join(folder_path, file2)
        else:
            # Fallback to looking for java files in the folder
            if folder_name:
                folder_path = os.path.join(base_path, folder_name)
                java_files = [f for f in os.listdir(folder_path) if f.endswith('.java')]
                if len(java_files) >= 2:
                    path1 = os.path.join(folder_path, java_files[0])
                    path2 = os.path.join(folder_path, java_files[1])
                    folder = folder_name
                else:
                    continue
            else:
                continue

        # Verify files exist
        if not (os.path.exists(path1) and os.path.exists(path2)):
            continue

        try:
            # Run TF-IDF detection
            tfidf_result = tfidf_detector.detect_plagiarism(path1, path2)
            tfidf_similarity = tfidf_result["similarity"]
            tfidf_prediction = 1 if tfidf_result["is_plagiarism"] else 0

            # Run AST-CC detection
            astcc_result = astcc_detector.detect_plagiarism(path1, path2)
            astcc_similarity = astcc_result.similarity_score
            astcc_prediction = 1 if astcc_similarity >= 0.45 else 0  # Adjust threshold as needed

            # Store results
            results.append({
                'folder': folder,
                'file1': os.path.basename(path1),
                'file2': os.path.basename(path2),
                'label': label,
                'dataset': dataset,
                'tfidf_similarity': tfidf_similarity,
                'astcc_similarity': astcc_similarity,
                'tfidf_prediction': tfidf_prediction,
                'astcc_prediction': astcc_prediction,
                'tfidf_correct': tfidf_prediction == label,
                'astcc_correct': astcc_prediction == label,
                'both_correct': (tfidf_prediction == label) and (astcc_prediction == label),
                'both_wrong': (tfidf_prediction != label) and (astcc_prediction != label)
            })

        except Exception as e:
            raise RuntimeError(f"Error processing files {path1} and {path2}: {e}")

    # Create DataFrame and calculate statistics
    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Save results to CSV
        results_df.to_csv(output_csv, index=False)

        # Print summary statistics
        print(f"\nComparison Summary:")
        print(f"  Total pairs analyzed: {len(results_df)}")

        tfidf_accuracy = results_df['tfidf_correct'].mean()
        astcc_accuracy = results_df['astcc_correct'].mean()

        print(f"  TF-IDF Accuracy: {tfidf_accuracy:.2%}")
        print(f"  AST-CC Accuracy: {astcc_accuracy:.2%}")
        print(f"  Both correct: {results_df['both_correct'].sum()} ({results_df['both_correct'].mean():.2%})")
        print(f"  Both wrong: {results_df['both_wrong'].sum()} ({results_df['both_wrong'].mean():.2%})")

        # Generate confusion matrices
        print("\nGenerating confusion matrices...")

        # TF-IDF confusion matrix
        plot_confusion_matrix(
            results_df['label'],
            results_df['tfidf_prediction'],
            "Matriz de Confusión - TF-IDF",
            "tfidf_confusion_matrix.png"
        )

        print("\nTF-IDF Report:")
        print(classification_report(
            results_df['label'],
            results_df['tfidf_prediction'],
            target_names=['No Plagio', 'Plagio']
        ))

        print("\nAST-CC Report:")
        print(classification_report(
            results_df['label'],
            results_df['astcc_prediction'],
            target_names=['No Plagio', 'Plagio']
        ))

        # AST-CC confusion matrix
        plot_confusion_matrix(
            results_df['label'],
            results_df['astcc_prediction'],
            "Matriz de Confusión - AST-CC",
            "astcc_confusion_matrix.png"
        )



        print("\nComparison results saved to:", output_csv)
    else:
        print("No valid file pairs found for analysis.")

    return results_df

if __name__ == "__main__":
    compare_plagiarism_detectors()
