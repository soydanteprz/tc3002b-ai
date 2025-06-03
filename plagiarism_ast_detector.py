import os
import pandas as pd
import javalang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re

class ASTNormalizer:
    """Normaliza código Java usando el AST para eliminar cambios superficiales"""

    def __init__(self):
        self.var_counter = 0
        self.method_counter = 0
        self.class_counter = 0
        self.param_counter = 0

        self.var_map = {}
        self.method_map = {}
        self.class_map = {}
        self.param_map = {}

    def reset_counters(self):
        """Reinicia los contadores para cada archivo"""
        self.var_counter = 0
        self.method_counter = 0
        self.class_counter = 0
        self.param_counter = 0

        self.var_map = {}
        self.method_map = {}
        self.class_map = {}
        self.param_map = {}

    def normalize_code(self, code):
        """Normaliza el código reemplazando identificadores"""
        try:
            self.reset_counters()
            tree = javalang.parse.parse(code)

            # Primero, recolectar todos los identificadores
            self._collect_identifiers(tree)

            # Luego, reemplazar en el código original
            normalized = self._replace_identifiers(code)

            return normalized
        except Exception as e:
            # Solo mostrar error si es significativo
            if "empty" not in str(e).lower():
                print(f"Error normalizando código: {e}")
            return code

    def _collect_identifiers(self, tree):
        """Recolecta todos los identificadores del AST"""
        # Clases
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            if node.name not in self.class_map:
                self.class_map[node.name] = f"CLASS{self.class_counter}"
                self.class_counter += 1

        # Métodos
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.name not in self.method_map and node.name != 'main':
                self.method_map[node.name] = f"METHOD{self.method_counter}"
                self.method_counter += 1

        # Parámetros
        for path, node in tree.filter(javalang.tree.FormalParameter):
            if node.name not in self.param_map:
                self.param_map[node.name] = f"PARAM{self.param_counter}"
                self.param_counter += 1

        # Variables locales
        for path, node in tree.filter(javalang.tree.LocalVariableDeclaration):
            for declarator in node.declarators:
                if declarator.name not in self.var_map:
                    self.var_map[declarator.name] = f"VAR{self.var_counter}"
                    self.var_counter += 1

        # Variables de campo
        for path, node in tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                if declarator.name not in self.var_map:
                    self.var_map[declarator.name] = f"FIELD{self.var_counter}"
                    self.var_counter += 1

    def _replace_identifiers(self, code):
        """Reemplaza los identificadores en el código"""
        # Crear un mapa combinado con prioridad
        all_mappings = {}
        all_mappings.update(self.var_map)
        all_mappings.update(self.param_map)
        all_mappings.update(self.method_map)
        all_mappings.update(self.class_map)

        # Ordenar por longitud descendente para evitar reemplazos parciales
        sorted_mappings = sorted(all_mappings.items(), key=lambda x: len(x[0]), reverse=True)

        normalized = code
        for original, replacement in sorted_mappings:
            # Usar word boundaries para evitar reemplazos parciales
            pattern = r'\b' + re.escape(original) + r'\b'
            normalized = re.sub(pattern, replacement, normalized)

        return normalized


class ASTStructureAnalyzer:
    """Analiza la estructura del AST para comparación"""

    @staticmethod
    def extract_ast_structure(code):
        """Extrae la secuencia de estructura del AST"""
        try:
            tree = javalang.parse.parse(code)
            structure = []

            # Recorrer el AST en profundidad
            for path, node in tree:
                node_type = type(node).__name__

                # Agregar información adicional según el tipo de nodo
                if isinstance(node, javalang.tree.MethodDeclaration):
                    # Incluir número de parámetros
                    param_count = len(node.parameters) if node.parameters else 0
                    structure.append(f"{node_type}_{param_count}")

                elif isinstance(node, javalang.tree.ForStatement):
                    structure.append(f"{node_type}_LOOP")

                elif isinstance(node, javalang.tree.WhileStatement):
                    structure.append(f"{node_type}_LOOP")

                elif isinstance(node, javalang.tree.IfStatement):
                    # Verificar si tiene else
                    has_else = 1 if node.else_statement else 0
                    structure.append(f"{node_type}_{has_else}")

                elif isinstance(node, javalang.tree.BinaryOperation):
                    structure.append(f"{node_type}_{node.operator}")

                else:
                    structure.append(node_type)

            return structure

        except Exception as e:
            # Solo mostrar error si es significativo
            if "empty" not in str(e).lower() and "NoneType" not in str(e):
                print(f"Error extrayendo estructura AST: {e}")
            return []

    @staticmethod
    def extract_method_signatures(code):
        """Extrae las firmas de los métodos normalizadas"""
        signatures = []
        try:
            tree = javalang.parse.parse(code)

            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                # Crear firma normalizada
                param_types = []
                if node.parameters:
                    for param in node.parameters:
                        param_type = param.type.name if hasattr(param.type, 'name') else str(param.type)
                        param_types.append(param_type)

                signature = f"METHOD({','.join(param_types)})"
                signatures.append(signature)

        except:
            pass

        return signatures

    @staticmethod
    def calculate_lcs(seq1, seq2):
        """Calcula la subsecuencia común más larga"""
        m, n = len(seq1), len(seq2)

        # Crear matriz DP
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Llenar la matriz
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    @staticmethod
    def structure_similarity(code1, code2):
        """Calcula la similitud estructural entre dos códigos"""
        struct1 = ASTStructureAnalyzer.extract_ast_structure(code1)
        struct2 = ASTStructureAnalyzer.extract_ast_structure(code2)

        if not struct1 or not struct2:
            return 0.0

        # LCS normalizado
        lcs_length = ASTStructureAnalyzer.calculate_lcs(struct1, struct2)
        similarity = 2 * lcs_length / (len(struct1) + len(struct2))

        return similarity

    @staticmethod
    def method_signature_similarity(code1, code2):
        """Compara las firmas de métodos entre dos códigos"""
        sigs1 = set(ASTStructureAnalyzer.extract_method_signatures(code1))
        sigs2 = set(ASTStructureAnalyzer.extract_method_signatures(code2))

        if not sigs1 and not sigs2:
            return 1.0  # Ambos sin métodos
        if not sigs1 or not sigs2:
            return 0.0

        # Jaccard similarity
        intersection = len(sigs1 & sigs2)
        union = len(sigs1 | sigs2)

        return intersection / union if union > 0 else 0.0


class EnhancedPlagiarismDetector:
    """Detector de plagio mejorado con normalización AST"""

    def __init__(self):
        self.normalizer = ASTNormalizer()
        self.structure_analyzer = ASTStructureAnalyzer()

    def read_java_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def clean_code(self, code):
        # Eliminar comentarios
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Eliminar espacios excesivos
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def tokenize_code(self, code):
        try:
            tokens = list(javalang.tokenizer.tokenize(code))
            return ' '.join(token.value for token in tokens)
        except:
            return ''

    def calculate_similarity(self, file1_path, file2_path):
        """Calcula múltiples métricas de similitud"""
        # Leer archivos
        code1_raw = self.read_java_file(file1_path)
        code2_raw = self.read_java_file(file2_path)

        # Limpiar código
        code1_clean = self.clean_code(code1_raw)
        code2_clean = self.clean_code(code2_raw)

        # Normalizar con AST
        code1_normalized = self.normalizer.normalize_code(code1_clean)
        code2_normalized = self.normalizer.normalize_code(code2_clean)

        # Tokenizar para TF-IDF
        tokens1 = self.tokenize_code(code1_normalized)
        tokens2 = self.tokenize_code(code2_normalized)

        results = {}

        # 1. TF-IDF con código normalizado
        if tokens1 and tokens2:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([tokens1, tokens2])
            results['tfidf_normalized'] = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        else:
            results['tfidf_normalized'] = 0.0

        # 2. Similitud estructural AST
        results['ast_structure'] = self.structure_analyzer.structure_similarity(code1_clean, code2_clean)

        # 3. Similitud de firmas de métodos
        results['method_signatures'] = self.structure_analyzer.method_signature_similarity(code1_clean, code2_clean)

        # 4. Similitud de secuencia (sobre código normalizado)
        results['sequence_normalized'] = SequenceMatcher(None, code1_normalized, code2_normalized).ratio()

        # 5. Puntuación combinada (ponderada)
        weights = {
            'tfidf_normalized': 0.35,
            'ast_structure': 0.30,
            'method_signatures': 0.20,
            'sequence_normalized': 0.15
        }

        results['combined_score'] = sum(results[metric] * weights[metric] for metric in weights)

        return results

    def process_dataset(self, base_path, csv_path, output_csv='similarity_ast_enhanced.csv', verbose=False):
        """Procesa el dataset completo con las nuevas métricas"""
        df = pd.read_csv(csv_path)
        df = df[df['source_dataset'].isin(['ir_plag', 'conplag'])]

        resultados = []
        total_files = len(df)
        print(f"\n🔍 Procesando {total_files} pares de archivos...")

        for idx, row in df.iterrows():
            # Solo mostrar progreso si verbose=True
            if verbose and idx % 50 == 0 and idx > 0:
                print(f"  ▶ {idx}/{total_files} completados ({idx/total_files*100:.1f}%)")

            dataset = row['source_dataset']
            plagio = row['label']
            file1 = row['file1']
            file2 = row['file2']
            file1_base = os.path.splitext(file1)[0]
            file2_base = os.path.splitext(file2)[0]

            # Construir paths según el dataset
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
                    print(f"[❌] Carpeta no encontrada para {file1} y {file2}")
                    continue

                path1 = os.path.join(folder_path, file1)
                path2 = os.path.join(folder_path, file2)

            else:
                continue

            if not (os.path.exists(path1) and os.path.exists(path2)):
                print(f"[⚠️] Archivos no encontrados: {path1}, {path2}")
                continue

            try:
                # Calcular todas las métricas
                metrics = self.calculate_similarity(path1, path2)

                resultado = {
                    'folder': os.path.basename(folder_path),
                    'file1': os.path.basename(path1),
                    'file2': os.path.basename(path2),
                    'es_plagio': plagio,
                    'dataset': dataset,
                    **metrics  # Agregar todas las métricas
                }

                resultados.append(resultado)

            except Exception as e:
                # Solo mostrar error si verbose=True
                if verbose:
                    print(f"[❌] Error procesando {folder_path}: {e}")
                continue

        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(output_csv, index=False)

        print(f"\n✅ Procesamiento completado: {len(resultados)}/{total_files} archivos")

        # Mostrar estadísticas
        print("\n📊 Estadísticas de resultados:")
        print("-" * 60)

        # Estadísticas por dataset y tipo
        stats = df_resultados.groupby(['dataset', 'es_plagio'])['combined_score'].agg(['mean', 'std', 'count'])
        print(stats.round(3))

        # Estadísticas generales
        print("\n📈 Resumen general:")
        print(f"  • Total de archivos procesados: {len(df_resultados)}")
        print(f"  • Casos de plagio: {df_resultados['es_plagio'].sum()} ({df_resultados['es_plagio'].mean()*100:.1f}%)")
        print(f"  • Score promedio (plagio=1): {df_resultados[df_resultados['es_plagio']==1]['combined_score'].mean():.3f}")
        print(f"  • Score promedio (plagio=0): {df_resultados[df_resultados['es_plagio']==0]['combined_score'].mean():.3f}")

        # Análisis de separación
        plagiarized_scores = df_resultados[df_resultados['es_plagio']==1]['combined_score']
        not_plagiarized_scores = df_resultados[df_resultados['es_plagio']==0]['combined_score']

        # Calcular overlap
        threshold = 0.5
        false_positives = (not_plagiarized_scores > threshold).sum()
        false_negatives = (plagiarized_scores < threshold).sum()

        print(f"\n🎯 Análisis con umbral={threshold}:")
        print(f"  • Falsos positivos: {false_positives} ({false_positives/len(not_plagiarized_scores)*100:.1f}%)")
        print(f"  • Falsos negativos: {false_negatives} ({false_negatives/len(plagiarized_scores)*100:.1f}%)")

        print(f"\n💾 Resultados guardados en: {output_csv}")

        return df_resultados


# Función principal para ejecutar
if __name__ == '__main__':
    BASE_PATH = 'data/splits/train'
    CSV_PATH = 'data/splits/train.csv'

    print("🚀 DETECTOR DE PLAGIO CON NORMALIZACIÓN AST")
    print("=" * 60)

    detector = EnhancedPlagiarismDetector()
    df_results = detector.process_dataset(BASE_PATH, CSV_PATH)

    # Análisis adicional de métricas individuales
    print("\n🔍 Análisis detallado de métricas:")
    print("-" * 60)
    metrics_cols = ['tfidf_normalized', 'ast_structure', 'method_signatures', 'sequence_normalized', 'combined_score']

    metrics_analysis = []
    for metric in metrics_cols:
        plag_mean = df_results[df_results['es_plagio']==1][metric].mean()
        no_plag_mean = df_results[df_results['es_plagio']==0][metric].mean()
        separation = plag_mean - no_plag_mean

        metrics_analysis.append({
            'Métrica': metric,
            'Media (Plagio)': round(plag_mean, 3),
            'Media (No Plagio)': round(no_plag_mean, 3),
            'Separación': round(separation, 3)
        })

    metrics_df = pd.DataFrame(metrics_analysis)
    print(metrics_df.to_string(index=False))

    # Identificar mejores métricas
    best_metric = metrics_df.loc[metrics_df['Separación'].idxmax()]
    print(f"\n⭐ Mejor métrica individual: {best_metric['Métrica']} (separación: {best_metric['Separación']})")

    print("\n✅ Análisis completado!")
    print("\n💡 Sugerencias:")
    print("  1. Ejecuta el análisis de visualización para gráficos detallados")
    print("  2. Considera ajustar los pesos de las métricas según tu dataset")
    print("  3. Prueba con diferentes umbrales para optimizar precision/recall")
