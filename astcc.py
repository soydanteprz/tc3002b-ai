"""
Implementación del algoritmo AST-CC (AST-based Code Comparison)
basado en el paper "An AST-Based Code Plagiarism Detection Algorithm"
de Jingling Zhao et al.
"""

import os
import javalang
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re



@dataclass
class NodeInfo:
    """Información de un nodo del AST"""
    hash_value: int
    node_type: str
    position: int
    line_number: int
    parent_hash: Optional[int] = None
    num_children: int = 0
    source_file: str = ""


@dataclass
class SimilarityResult:
    """Resultado de similitud entre dos archivos"""
    file1: str
    file2: str
    similar_nodes: List[Tuple[NodeInfo, NodeInfo]]
    similarity_score: float


class ASTCCDetector:
    """
    Implementación del algoritmo AST-CC para detección de plagio
    """

    def __init__(self):
        # Pesos para operaciones especiales (para evitar falsos positivos)
        self.operation_weights = {
            '-': {'left': 1, 'right': 2},  # Resta
            '/': {'left': 1, 'right': 2},  # División
            '%': {'left': 1, 'right': 2},  # Módulo
        }

        # Valores hash base para diferentes tipos de nodos
        self.node_type_values = {
            'MethodDeclaration': 100,
            'ClassDeclaration': 200,
            'VariableDeclaration': 50,
            'BinaryOperation': 20,
            'Assignment': 30,
            'IfStatement': 40,
            'ForStatement': 60,
            'WhileStatement': 65,
            'FunctionCall': 20,
            'Parameter': 15,
            'Literal': 10
        }

    def _get_node_type_value(self, node) -> int:
        """Obtiene el valor base para un tipo de nodo"""
        node_type = type(node).__name__
        return self.node_type_values.get(node_type, 5)

    def _calculate_node_hash(self, node, parent_node=None) -> int:
        """
        Calcula el hash de un nodo según el algoritmo del paper
        """
        # Valor base del tipo de nodo
        base_value = self._get_node_type_value(node)

        # Para operaciones binarias, aplicar pesos especiales
        if isinstance(node, javalang.tree.BinaryOperation):
            operator = node.operator
            if operator in self.operation_weights:
                # Calcular hash considerando pesos diferentes para izquierda y derecha
                left_weight = self.operation_weights[operator]['left']
                right_weight = self.operation_weights[operator]['right']

                left_hash = self._calculate_subtree_hash(node.operandl) if hasattr(node, 'left') else 0
                right_hash = self._calculate_subtree_hash(node.operandr) if hasattr(node, 'right') else 0

                return base_value + left_weight * left_hash + right_weight * right_hash

        # Para otros nodos, sumar los hashes de los hijos
        child_hash_sum = 0
        for attr_name in node.attrs:
            attr_value = getattr(node, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, javalang.tree.Node):
                            child_hash_sum += self._calculate_subtree_hash(item)
                elif isinstance(attr_value, javalang.tree.Node):
                    child_hash_sum += self._calculate_subtree_hash(attr_value)

        return base_value + child_hash_sum

    def _calculate_subtree_hash(self, node) -> int:
        """Calcula el hash de un subárbol completo"""
        if node is None:
            return 0

        return self._calculate_node_hash(node)

    def _count_children(self, node) -> int:
        """Cuenta el número de hijos directos de un nodo"""
        count = 0
        for attr_name in node.attrs:
            attr_value = getattr(node, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, list):
                    count += sum(1 for item in attr_value if isinstance(item, javalang.tree.Node))
                elif isinstance(attr_value, javalang.tree.Node):
                    count += 1
        return count

    def extract_ast_info(self, code: str, filename: str = "") -> Dict[int, List[NodeInfo]]:
        """
        Extrae información del AST y la organiza por número de sub-nodos
        Retorna: HashListArray donde el índice es el número de sub-nodos
        """
        hash_list_array = defaultdict(list)

        try:
            tree = javalang.parse.parse(code)

            # Recorrer el AST
            position = 0
            for path, node in tree:
                # Calcular hash del nodo
                node_hash = self._calculate_node_hash(node)

                # Contar hijos
                num_children = self._count_children(node)

                # Obtener línea (si está disponible)
                line_number = 0
                if hasattr(node, 'position') and node.position:
                    line_number = node.position.line

                # Crear información del nodo
                node_info = NodeInfo(
                    hash_value=node_hash,
                    node_type=type(node).__name__,
                    position=position,
                    line_number=line_number,
                    num_children=num_children,
                    source_file=filename
                )

                # Almacenar en el array según el número de hijos
                hash_list_array[num_children].append(node_info)
                position += 1

        except Exception as e:
            print(f"Error procesando {filename}: {e}")

        return hash_list_array

    def compare_hash_arrays(self, suspected_array: Dict[int, List[NodeInfo]],
                            original_array: Dict[int, List[NodeInfo]]) -> List[Tuple[NodeInfo, NodeInfo]]:
        """
        Compara dos arrays de hash según el algoritmo del paper
        """
        similar_nodes = []

        # Comparar solo nodos con el mismo número de hijos
        for num_children in suspected_array.keys():
            if num_children not in original_array:
                continue

            suspected_nodes = sorted(suspected_array[num_children],
                                     key=lambda x: (x.hash_value, x.position))
            original_nodes = sorted(original_array[num_children],
                                    key=lambda x: (x.hash_value, x.position))

            # Algoritmo de comparación del paper
            i, j = 0, 0

            while i < len(suspected_nodes) and j < len(original_nodes):
                susp_node = suspected_nodes[i]
                orig_node = original_nodes[j]

                # Si el nodo raíz es igual, todos los nodos son similares
                if num_children == 0 and susp_node.hash_value == orig_node.hash_value:
                    similar_nodes.append((susp_node, orig_node))
                    break

                # Verificar si los padres ya fueron comparados (optimización)
                if susp_node.parent_hash and orig_node.parent_hash:
                    if susp_node.parent_hash == orig_node.parent_hash:
                        i += 1
                        j += 1
                        continue

                # Comparar valores hash
                if susp_node.hash_value < orig_node.hash_value:
                    i += 1
                elif susp_node.hash_value > orig_node.hash_value:
                    j += 1
                else:
                    # Hashes iguales - nodos similares
                    similar_nodes.append((susp_node, orig_node))
                    i += 1
                    j += 1

        return similar_nodes

    def detect_plagiarism(self, suspected_file: str, original_file: str) -> SimilarityResult:
        """
        Detecta plagio entre dos archivos usando el algoritmo AST-CC
        """
        # Leer archivos
        with open(suspected_file, 'r', encoding='utf-8') as f:
            suspected_code = f.read()
        with open(original_file, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # Extraer información del AST
        suspected_array = self.extract_ast_info(suspected_code, suspected_file)
        original_array = self.extract_ast_info(original_code, original_file)

        # Comparar arrays
        similar_nodes = self.compare_hash_arrays(suspected_array, original_array)

        # Calcular score de similitud
        total_suspected = sum(len(nodes) for nodes in suspected_array.values())
        total_original = sum(len(nodes) for nodes in original_array.values())

        if total_suspected == 0 or total_original == 0:
            similarity_score = 0.0
        else:
            similarity_score = 2 * len(similar_nodes) / (total_suspected + total_original)

        return SimilarityResult(
            file1=suspected_file,
            file2=original_file,
            similar_nodes=similar_nodes,
            similarity_score=similarity_score
        )

    def process_dataset_astcc(self, base_path: str, csv_path: str, output_csv: str = 'astcc_results.csv'):
        """
        Procesa el dataset usando el algoritmo AST-CC
        """
        df = pd.read_csv(csv_path)
        df = df[df['source_dataset'].isin(['ir_plag', 'conplag'])]

        results = []
        total = len(df)

        print(f"\n🚀 Procesando {total} pares con algoritmo AST-CC...")

        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                print(f"  ▶ {idx}/{total} completados ({idx/total*100:.1f}%)")

            dataset = row['source_dataset']
            plagio = row['label']
            file1 = row['file1']
            file2 = row['file2']
            file1_base = os.path.splitext(file1)[0]
            file2_base = os.path.splitext(file2)[0]

            # Construir paths
            if dataset == 'ir_plag':
                folder = row['folder_name']
                folder_path = os.path.join(base_path, folder)
                path1 = os.path.join(folder_path, 'original.java')
                path2 = os.path.join(folder_path, 'compared.java')
            elif dataset == 'conplag':
                folder1 = os.path.join(base_path, f"{file1_base}_{file2_base}")
                folder2 = os.path.join(base_path, f"{file2_base}_{file1_base}")

                if os.path.isdir(folder1):
                    folder_path = folder1
                elif os.path.isdir(folder2):
                    folder_path = folder2
                else:
                    continue

                path1 = os.path.join(folder_path, file1)
                path2 = os.path.join(folder_path, file2)
            else:
                continue

            if not (os.path.exists(path1) and os.path.exists(path2)):
                continue

            try:
                # Detectar plagio con AST-CC
                result = self.detect_plagiarism(path1, path2)

                results.append({
                    'folder': os.path.basename(folder_path),
                    'file1': os.path.basename(path1),
                    'file2': os.path.basename(path2),
                    'es_plagio': plagio,
                    'dataset': dataset,
                    'astcc_score': result.similarity_score,
                    'similar_nodes': len(result.similar_nodes)
                })

            except Exception as e:
                print(f"Error procesando {folder_path}: {e}")
                continue

        # Guardar resultados
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)

        print(f"\nProcesamiento completado: {len(results)}/{total} archivos")

        # Estadísticas
        print("\nEstadísticas AST-CC:")
        print("-" * 50)

        if len(df_results) > 0:
            stats = df_results.groupby(['dataset', 'es_plagio'])['astcc_score'].agg(['mean', 'std', 'count'])
            print(stats.round(3))

            print(f"\n📈 Resumen:")
            print(f"  • Score promedio (plagio=1): {df_results[df_results['es_plagio']==1]['astcc_score'].mean():.3f}")
            print(f"  • Score promedio (plagio=0): {df_results[df_results['es_plagio']==0]['astcc_score'].mean():.3f}")

            # Análisis con umbral
            threshold = 0.55  # Umbral típico para AST-CC
            tp = ((df_results['astcc_score'] >= threshold) & (df_results['es_plagio'] == 1)).sum()
            tn = ((df_results['astcc_score'] < threshold) & (df_results['es_plagio'] == 0)).sum()
            fp = ((df_results['astcc_score'] >= threshold) & (df_results['es_plagio'] == 0)).sum()
            fn = ((df_results['astcc_score'] < threshold) & (df_results['es_plagio'] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n🎯 Métricas con umbral={threshold}:")
            print(f"  • Precision: {precision:.3f}")
            print(f"  • Recall: {recall:.3f}")
            print(f"  • F1-Score: {f1:.3f}")

        print(f"\n💾 Resultados guardados en: {output_csv}")

        return df_results

class TFIDFPlagiarismDetector:
    """Detector that uses TF-IDF to identify code plagiarism"""

    def __init__(self, threshold=0.55):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 3),
            stop_words=None
        )

    def read_file(self, filepath):
        """Read a Java file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""

    def preprocess_code(self, code):
        """Clean and normalize Java code for better comparison"""
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code).strip()

        # Normalize variable names
        try:
            # Use javalang to tokenize the code
            tokens = list(javalang.tokenizer.tokenize(code))
            return ' '.join(token.value for token in tokens)
        except:
            # If tokenization fails, return cleaned code
            return code

    def calculate_similarity(self, text1, text2):
        """Calculate TF-IDF similarity between two text samples"""
        if not text1 or not text2:
            return 0.0

        try:
            # Transform texts to TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def detect_plagiarism(self, file1_path, file2_path):
        """Detect if two files are plagiarized"""
        # Read files
        code1 = self.read_file(file1_path)
        code2 = self.read_file(file2_path)

        # Preprocess code
        processed1 = self.preprocess_code(code1)
        processed2 = self.preprocess_code(code2)

        # Calculate similarity
        similarity = self.calculate_similarity(processed1, processed2)

        # Determine if plagiarized
        is_plagiarism = similarity >= self.threshold

        return {
            "similarity": similarity,
            "is_plagiarism": is_plagiarism
        }


