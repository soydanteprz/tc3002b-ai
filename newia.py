from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple, Dict
import os
import pickle
from astcc import ASTCCDetector, TFIDFPlagiarismDetector
from visualization_tools import *


class ASTCCFeatureVectorizer:
    """
    Processes Abstract Syntax Tree Cross Comparison (ASTCC) features for given
    code pairs by extracting, transforming, and analyzing AST-based structural
    similarities and differences.

    This class is primarily used for comparing the ASTs of two code snippets,
    identifying similarities and divergences across their respective structures,
    and generating a feature vector that quantifies these relationships. These
    features are essential in tasks such as code similarity analysis, plagiarism
    detection, or structural analysis of source code.

    :ivar astcc_detector: An instance of ASTCCDetector used for AST extraction
        and analysis.
    :type astcc_detector: ASTCCDetector
    :ivar feature_names: A list of names of the features extracted and used in
        feature vectors.
    :type feature_names: list of str
    :ivar is_fitted: A boolean that indicates whether the vectorizer has been
        fitted using the input data.
    :type is_fitted: bool
    """

    def __init__(self):
        """
        Initializes the ASTCCFeatureVectorizer with an instance of ASTCCDetector
        and prepares an empty list for feature names. The `is_fitted` flag is set
        to `False` to indicate that the vectorizer has not yet been fitted with
        any data.
        :ivar astcc_detector: An instance of ASTCCDetector for AST extraction
            and analysis.
        :type astcc_detector: ASTCCDetector
        :ivar feature_names: A list of names of the features extracted and used
            in feature vectors.
        :type feature_names: list of str
        :ivar is_fitted: A boolean that indicates whether the vectorizer has been
            fitted using the input data.
        :type is_fitted: bool
        """
        self.astcc_detector = ASTCCDetector()
        self.feature_names = []
        self.is_fitted = False

    def _extract_astcc_features(self, code1: str, code2: str) -> np.ndarray:
        """
        Extracts and calculates AST code comparison features between two code samples.

        This function extracts Abstract Syntax Tree (AST) features for two given code inputs,
        compares their structural and syntactical similarities, and computes a feature set
        representing these similarities and differences. The feature set includes metrics such
        as node similarity scores, node count differences, ratios, distribution diversity, and
        structural complexity metrics.

        :param code1: The first code sample in string format.
        :type code1: str
        :param code2: The second code sample in string format.
        :type code2: str
        :return: A numpy array containing calculated feature values derived from AST-based comparisons.
        :rtype: np.ndarray
        """
        # 1. AST extraction
        array1 = self.astcc_detector.extract_ast_info(code1, "code1")
        array2 = self.astcc_detector.extract_ast_info(code2, "code2")

        # 2. Encontrar nodos similares
        similar_nodes = self.astcc_detector.compare_hash_arrays(array1, array2)

        # 3. Calcular características derivadas
        features = {}

        # Características básicas de similitud
        total_nodes1 = sum(len(nodes) for nodes in array1.values())
        total_nodes2 = sum(len(nodes) for nodes in array2.values())

        features['astcc_similarity_score'] = (
            2 * len(similar_nodes) / (total_nodes1 + total_nodes2)
            if (total_nodes1 + total_nodes2) > 0 else 0
        )
        features['astcc_similar_nodes_count'] = len(similar_nodes)
        features['astcc_total_nodes_diff'] = abs(total_nodes1 - total_nodes2)
        features['astcc_nodes_ratio'] = (
            min(total_nodes1, total_nodes2) / max(total_nodes1, total_nodes2)
            if max(total_nodes1, total_nodes2) > 0 else 1.0
        )

        # Características por número de hijos (0 a 5)
        for num_children in range(6):
            nodes1 = len(array1.get(num_children, []))
            nodes2 = len(array2.get(num_children, []))

            features[f'astcc_nodes_{num_children}children_diff'] = abs(nodes1 - nodes2)
            features[f'astcc_nodes_{num_children}children_ratio'] = (
                min(nodes1, nodes2) / max(nodes1, nodes2)
                if max(nodes1, nodes2) > 0 else 1.0
            )
            features[f'astcc_nodes_{num_children}children_sim'] = (
                    1.0 - abs(nodes1 - nodes2) / max(nodes1 + nodes2, 1)
            )

        # Características por tipo de nodo
        node_type_stats1 = self._get_node_type_statistics(array1)
        node_type_stats2 = self._get_node_type_statistics(array2)

        for node_type in ['MethodDeclaration', 'ClassDeclaration', 'VariableDeclaration',
                          'BinaryOperation', 'Assignment', 'IfStatement', 'ForStatement',
                          'WhileStatement', 'FunctionCall', 'Parameter', 'Literal']:
            count1 = node_type_stats1.get(node_type, 0)
            count2 = node_type_stats2.get(node_type, 0)

            features[f'astcc_{node_type.lower()}_diff'] = abs(count1 - count2)
            features[f'astcc_{node_type.lower()}_ratio'] = (
                min(count1, count2) / max(count1, count2)
                if max(count1, count2) > 0 else 1.0
            )
            features[f'astcc_{node_type.lower()}_sim'] = (
                    1.0 - abs(count1 - count2) / max(count1 + count2, 1)
            )

        # Características avanzadas de distribución de hash
        all_hashes1 = [node.hash_value for nodes in array1.values() for node in nodes]
        all_hashes2 = [node.hash_value for nodes in array2.values() for node in nodes]

        features['astcc_unique_hashes1'] = len(set(all_hashes1))
        features['astcc_unique_hashes2'] = len(set(all_hashes2))
        features['astcc_hash_diversity_diff'] = abs(
            features['astcc_unique_hashes1'] - features['astcc_unique_hashes2']
        )
        features['astcc_hash_diversity_ratio'] = (
            min(features['astcc_unique_hashes1'], features['astcc_unique_hashes2']) /
            max(features['astcc_unique_hashes1'], features['astcc_unique_hashes2'])
            if max(features['astcc_unique_hashes1'], features['astcc_unique_hashes2']) > 0 else 1.0
        )

        # Características de complejidad estructural
        features['astcc_avg_hash_value1'] = np.mean(all_hashes1) if all_hashes1 else 0
        features['astcc_avg_hash_value2'] = np.mean(all_hashes2) if all_hashes2 else 0
        features['astcc_hash_value_diff'] = abs(
            features['astcc_avg_hash_value1'] - features['astcc_avg_hash_value2']
        )

        # Distribución de profundidad
        max_children1 = max(array1.keys()) if array1 else 0
        max_children2 = max(array2.keys()) if array2 else 0
        features['astcc_max_depth_diff'] = abs(max_children1 - max_children2)
        features['astcc_structural_complexity_sim'] = (
                1.0 - abs(max_children1 - max_children2) / max(max_children1 + max_children2, 1)
        )

        return np.array(list(features.values()))

    def _get_node_type_statistics(self, ast_array) -> dict:
        """
        Computes the frequency statistics of each node type found in the given
        abstract syntax tree (AST) array. This method iterates through the supplied
        AST nodes and collects a count for each unique node type, summarizing
        the distribution of node types in the returned statistics dictionary.

        :param ast_array: A dictionary containing AST nodes, where the keys
            are identifiers (e.g., file names, module names) and the values are
            lists of nodes associated with each key. Each node is expected to
            have a `node_type` attribute.
        :type ast_array: dict

        :return: A dictionary mapping each unique node type found in the provided
            `ast_array` to its frequency count across all nodes. The keys in the
            returned dictionary represent node types, and the values are the
            respective counts of occurrences for each node type.
        :rtype: dict
        """
        stats = {}
        for nodes in ast_array.values():
            for node in nodes:
                node_type = node.node_type
                if node_type not in stats:
                    stats[node_type] = 0
                stats[node_type] += 1
        return stats

    def get_feature_names(self) -> List[str]:
        """
        Generates a list of feature names based on specific criteria, such as node
        count, type of node, number of children, and other advanced characteristics.
        This function initializes the `feature_names` attribute with a predefined set
        of feature identifiers if it has not already been established. These feature
        names are categorized into basic features, features based on the number of
        children, features categorized by node type, and advanced features. The
        returned list can be used for analytical or comparative purposes related to
        abstract syntax tree (AST) structures.

        :raises AttributeError: If the `self.feature_names` attribute cannot be accessed
            on the instance (e.g., missing or improperly defined).
        :return: A list of feature names used for AST-related evaluations.
        :rtype: List[str]
        """
        if not self.feature_names:
            # Características básicas
            names = ['astcc_similarity_score', 'astcc_similar_nodes_count',
                     'astcc_total_nodes_diff', 'astcc_nodes_ratio']

            # Por número de hijos
            for num_children in range(6):
                names.extend([
                    f'astcc_nodes_{num_children}children_diff',
                    f'astcc_nodes_{num_children}children_ratio',
                    f'astcc_nodes_{num_children}children_sim'
                ])

            # Por tipo de nodo
            for node_type in ['MethodDeclaration', 'ClassDeclaration', 'VariableDeclaration',
                              'BinaryOperation', 'Assignment', 'IfStatement', 'ForStatement',
                              'WhileStatement', 'FunctionCall', 'Parameter', 'Literal']:
                names.extend([
                    f'astcc_{node_type.lower()}_diff',
                    f'astcc_{node_type.lower()}_ratio',
                    f'astcc_{node_type.lower()}_sim'
                ])

            # Características avanzadas
            names.extend([
                'astcc_unique_hashes1', 'astcc_unique_hashes2', 'astcc_hash_diversity_diff',
                'astcc_hash_diversity_ratio', 'astcc_avg_hash_value1', 'astcc_avg_hash_value2',
                'astcc_hash_value_diff', 'astcc_max_depth_diff', 'astcc_structural_complexity_sim'
            ])

            self.feature_names = names

        return self.feature_names

    def fit_transform(self, code_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Transform the given pairs of code strings into a numerical numpy array
        and mark the instance as fitted. This method is intended to fit the
        transformer on given code pairs and then transform them.

        :param code_pairs: A list of tuples where each tuple contains two strings
            representing pairs of code fragments
        :return: A numpy array representing the transformed numerical representation
            of the provided code pairs
        """
        self.is_fitted = True
        return self.transform(code_pairs)

    def transform(self, code_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Transforms a list of code pairs into a feature matrix by extracting ASTCC features
        for each code pair. This method processes each code pair by invoking an internal
        feature extraction method and aggregates the results in a matrix form. Errors
        in processing a code pair result in appending a zero-filled feature vector for
        that pair.

        :param code_pairs: A list of tuples where each tuple contains two code fragments
                           (code1, code2) represented as strings.
        :type code_pairs: List[Tuple[str, str]]
        :return: A numpy array representing the extracted feature matrix. Each row
                 corresponds to a feature vector for a code pair.
        :rtype: np.ndarray
        """
        features_matrix = []

        print(f"Extracting ASTCC features for {len(code_pairs)} code pairs...")

        for i, (code1, code2) in enumerate(code_pairs):
            if i % 100 == 0 and i > 0:
                print(f"  ⚡ Processed {i}/{len(code_pairs)} pairs ({i/len(code_pairs)*100:.1f}%)...")

            try:
                features = self._extract_astcc_features(code1, code2)
                features_matrix.append(features)
            except Exception as e:
                print(f"  ⚠️ Error processing pair {i}: {e}")
                features_matrix.append(np.zeros(len(self.get_feature_names())))

        print(f"✅ ASTCC feature extraction completed! Shape: {np.array(features_matrix).shape}")
        return np.array(features_matrix)


class HybridPlagiarismClassifier:
    """
    HybridPlagiarismClassifier is designed to detect plagiarism in code pairs by combining
    features extracted using two methods: TF-IDF and ASTCC. The class provides functionality
    to train a hybrid model, predict plagiarism likelihood, and evaluate model performance.

    This classifier uses logistic regression and allows adjustable weighting between TF-IDF
    and ASTCC feature contributions. It also supports feature scaling, selection, and preprocessing.

    :ivar use_tfidf: Indicates if TF-IDF-based features are enabled.
    :ivar use_astcc: Indicates if ASTCC-based features are enabled.
    :ivar astcc_weight: Weight assigned to ASTCC features during combination.
    :ivar tfidf_weight: Weight assigned to TF-IDF features during combination.
    :ivar astcc_vectorizer: Component used to extract ASTCC features.
    :ivar tfidf_detector: Component used to extract and preprocess TF-IDF features.
    :ivar model: Logistic regression model for classification.
    :ivar feature_selector: Feature selection component to reduce feature dimensions.
    :ivar scaler: Preprocessing component to scale feature values.
    :ivar is_trained: Indicates whether the model has been trained.
    :ivar feature_names: Names of all available features used in training and prediction.
    """

    def __init__(self, use_tfidf=True, use_astcc=True,
                 astcc_weight=0.5, tfidf_weight=0.5):
        """
        Initializes the configuration and components for a hybrid text detection system
        using ASTCC and/or TF-IDF features with Logistic Regression as the underlying
        classification model. Adjusts weights automatically based on provided
        configurations and initializes feature preprocessing tools.

        :param use_tfidf: Indicates whether to enable the TF-IDF feature detection.
        :type use_tfidf: bool

        :param use_astcc: Indicates whether to enable the ASTCC feature detection.
        :type use_astcc: bool

        :param astcc_weight: Weight assigned to the ASTCC features if hybrid mode is used.
        :type astcc_weight: float

        :param tfidf_weight: Weight assigned to the TF-IDF features if hybrid mode is used.
        :type tfidf_weight: float

        :raises ValueError: Raised when no feature is enabled or when total weight from
            astcc_weight and tfidf_weight is zero in hybrid configuration.
        """
        self.use_tfidf = use_tfidf
        self.use_astcc = use_astcc

        # Ajuste automático de pesos
        if not use_tfidf and use_astcc:
            self.astcc_weight = 1.0
            self.tfidf_weight = 0.0
            print("Configuration: Only ASTCC")
        elif use_tfidf and not use_astcc:
            self.astcc_weight = 0.0
            self.tfidf_weight = 1.0
            print("Configuration: Only TF-IDF")
        elif use_tfidf and use_astcc:
            total_weight = astcc_weight + tfidf_weight
            if total_weight == 0:
                raise ValueError("Total weight for hybrid configuration cannot be zero")
            self.astcc_weight = astcc_weight / total_weight
            self.tfidf_weight = tfidf_weight / total_weight
            print(f"Configuration: Hybrid (ASTCC:{self.astcc_weight:.1f}, TF-IDF:{self.tfidf_weight:.1f})")
        else:
            raise ValueError("At least one feature must be enabled (TF-IDF or ASTCC)")

        # Componentes
        if self.use_astcc:
            self.astcc_vectorizer = ASTCCFeatureVectorizer()

        if self.use_tfidf:
            self.tfidf_detector = TFIDFPlagiarismDetector()
            self.tfidf_detector.vectorizer.ngram_range = (1, 3)
            self.tfidf_detector.vectorizer.max_features = 5000

        # Modelo y preprocesamiento
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.scaler = StandardScaler()

        self.is_trained = False
        self.feature_names = []

    def _extract_tfidf_features(self, code_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Extracts TF-IDF features from given pairs of code snippets.

        This method processes pairs of code snippets to create combined text sequences,
        where each pair is concatenated with a separator token "[SEP]". These
        combined sequences are then transformed into a TF-IDF feature matrix using a
        fitted TF-IDF vectorizer. If the vectorizer has not been previously fitted,
        it will be trained on the combined sequences during this process.

        :param code_pairs: A list of tuples, where each tuple contains two strings
                           representing pairs of code snippets to be processed.
        :type code_pairs: List[Tuple[str, str]]
        :return: A 2D NumPy array where rows correspond to the processed TF-IDF
                 feature vectors for the input code pairs.
        :rtype: np.ndarray
        """
        combined_texts = []

        for code1, code2 in code_pairs:
            processed1 = self.tfidf_detector.preprocess_code(code1)
            processed2 = self.tfidf_detector.preprocess_code(code2)
            combined_text = processed1 + " [SEP] " + processed2
            combined_texts.append(combined_text)

        if not hasattr(self.tfidf_detector.vectorizer, 'vocabulary_'):
            tfidf_matrix = self.tfidf_detector.vectorizer.fit_transform(combined_texts)
        else:
            tfidf_matrix = self.tfidf_detector.vectorizer.transform(combined_texts)

        return tfidf_matrix.toarray()

    def _combine_features(self, tfidf_features=None, astcc_features=None) -> np.ndarray:
        """
        Combines TF-IDF and ASTCC features using specified weights for each feature
        set. If both feature sets are available, they are horizontally concatenated
        to form a single feature vector. If no features are provided, an exception
        is raised. This function is designed to work internally within the class,
        and its usage depends on the state of the class attributes `use_tfidf`,
        `use_astcc`, `tfidf_weight`, and `astcc_weight`.

        :param tfidf_features: A numpy array representing the TF-IDF features. If
            provided and `use_tfidf` is enabled in the instance, these features are
            included in the combined feature set.
        :param astcc_features: A numpy array representing the ASTCC features. If
            provided and `use_astcc` is enabled in the instance, these features
            are included in the combined feature set.
        :return: A single numpy array containing the concatenated feature vector
            when at least one feature set is available.
        :raises ValueError: If neither TF-IDF nor ASTCC features are provided.
        """
        combined_features = []

        if self.use_tfidf and tfidf_features is not None:
            tfidf_weighted = tfidf_features * self.tfidf_weight
            combined_features.append(tfidf_weighted)

        if self.use_astcc and astcc_features is not None:
            astcc_weighted = astcc_features * self.astcc_weight
            combined_features.append(astcc_weighted)

        if combined_features:
            return np.hstack(combined_features)
        else:
            raise ValueError("No features available")

    def _prepare_feature_names(self, tfidf_features=None, astcc_features=None):
        """
        Prepares a list of feature names based on the TF-IDF and ASTCC features enabled
        and available. This function takes in optional TF-IDF and ASTCC feature sets,
        and generates appropriate feature names accordingly.

        :param tfidf_features: Optional feature set for TF-IDF, allowing preparation of
            feature names if TF-IDF usage is enabled and the features are provided.
        :type tfidf_features: Optional[Any]
        :param astcc_features: Optional feature set for ASTCC, allowing preparation of
            feature names if ASTCC usage is enabled and the features are provided.
        :type astcc_features: Optional[Any]
        :return: A list of prepared feature names combining both TF-IDF and ASTCC based
            features if applicable. The list is empty if neither feature type is utilized.
        :rtype: List[str]
        """
        feature_names = []

        if self.use_tfidf and tfidf_features is not None:
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_detector.vectorizer.get_feature_names_out()]
            feature_names.extend(tfidf_names)

        if self.use_astcc and astcc_features is not None:
            astcc_names = [f"{name}" for name in self.astcc_vectorizer.get_feature_names()]
            feature_names.extend(astcc_names)

        return feature_names

    def train(self, code_pairs: List[Tuple[str, str]], labels: List[int]):
        """
        Trains a hybrid classification model by combining features extracted using TF-IDF and ASTCC,
        selecting the most informative features, normalizing the data, and fitting a logistic regression
        model. The training process involves multiple steps such as feature extraction, combination,
        selection, and normalization.

        :param code_pairs: A list of code pairs used as input data for feature extraction. Each code pair
            is represented as a tuple of strings.
        :param labels: A list of integer labels corresponding to each code pair. These labels represent
            the ground truth classification for the provided code pairs.
        :return: None
        """

        print(f"\nTraining hybrid model (TF-IDF: {self.use_tfidf}, ASTCC: {self.use_astcc})")
        print(f"Weights - ASTCC: {self.astcc_weight}, TF-IDF: {self.tfidf_weight}")

        # Extraer características
        tfidf_features = None
        if self.use_tfidf:
            print("Extracting TF-IDF features...")
            tfidf_features = self._extract_tfidf_features(code_pairs)
            print(f"TF-IDF features shape: {tfidf_features.shape}")

        astcc_features = None
        if self.use_astcc:
            print("Extracting ASTCC features...")
            astcc_features = self.astcc_vectorizer.fit_transform(code_pairs)
            print(f"ASTCC features shape: {astcc_features.shape}")

        # Combinar características
        X = self._combine_features(tfidf_features, astcc_features)
        print(f"Combined features shape: {X.shape}")

        # Preparar nombres de características
        self.feature_names = self._prepare_feature_names(tfidf_features, astcc_features)

        # Selección de características
        if X.shape[1] > 1000:
            self.feature_selector.k = min(1000, X.shape[1])
            X = self.feature_selector.fit_transform(X, labels)
            print(f"Selected {X.shape[1]} most important features")

        # Normalizar y entrenar
        X = self.scaler.fit_transform(X)
        print("Training logistic regression...")
        self.model.fit(X, labels)

        self.is_trained = True
        print("Hybrid model with ASTCC trained successfully!")

        # Mostrar estadísticas de entrenamiento
        train_pred = self.model.predict(X)
        train_accuracy = accuracy_score(labels, train_pred)
        print(f"Training accuracy: {train_accuracy:.3f}")

    def predict(self, code_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Predict outputs based on provided pairs of code snippets.

        This method processes the given pairs of code snippets by extracting features
        using TF-IDF and/or AST Contextual Clustering (ASTCC) if enabled, combining
        these features, applying feature selection, scaling the data, and finally using
        the trained model to predict the outputs.

        :param code_pairs: List of tuples where each tuple contains two code snippets
            in the form of strings to be analyzed and compared.
        :type code_pairs: List[Tuple[str, str]]
        :return: NumPy array containing the predictions made by the trained model for
            the input code pairs.
        :rtype: numpy.ndarray
        :raises ValueError: If the model has not been trained before calling this
            method.
        """

        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Extraer características
        tfidf_features = None
        if self.use_tfidf:
            tfidf_features = self._extract_tfidf_features(code_pairs)

        astcc_features = None
        if self.use_astcc:
            astcc_features = self.astcc_vectorizer.transform(code_pairs)


        X = self._combine_features(tfidf_features, astcc_features)

        if hasattr(self.feature_selector, 'scores_'):
            X = self.feature_selector.transform(X)

        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, code_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Predicts probabilities for the given code pairs using the trained model.

        The function computes predictions by extracting features, combining them,
        and applying preprocessing steps before generating probabilities for each code pair.
        It uses TF-IDF and ASTCC features if enabled, applies feature selection if a feature
        selector with scores is provided, and uses a scaler to normalize the data before
        passing it to the model.

        :param code_pairs: A list of tuples where each tuple contains two code strings
                          to be processed.
        :type code_pairs: List[Tuple[str, str]]
        :return: A numpy array containing the predicted probabilities for each code pair.
        :rtype: np.ndarray
        :raises ValueError: If the model is not trained before calling this method.
        """

        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Extraer características
        tfidf_features = None
        if self.use_tfidf:
            tfidf_features = self._extract_tfidf_features(code_pairs)

        astcc_features = None
        if self.use_astcc:
            astcc_features = self.astcc_vectorizer.transform(code_pairs)

        # Combinar y procesar
        X = self._combine_features(tfidf_features, astcc_features)

        if hasattr(self.feature_selector, 'scores_'):
            X = self.feature_selector.transform(X)

        X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def evaluate(self, code_pairs: List[Tuple[str, str]], labels: List[int], dataset_name: str = "Test") -> Dict:
        """
        Evaluates the performance of the model on given code pairs and their corresponding
        labels, computing metrics such as accuracy and AUC-ROC score.

        The method first generates predictions and probabilities using the defined
        `predict` and `predict_proba` methods. It calculates accuracy and AUC-ROC,
        and prints a detailed classification report for evaluation, including metrics
        for each target class. Finally, it returns a dictionary containing the computed
        metrics, predictions, and probabilities.

        :param code_pairs: List of code snippet pairs to evaluate.
        :type code_pairs: List[Tuple[str, str]]
        :param labels: List of ground truth labels where each label corresponds to a code pair.
        :type labels: List[int]
        :param dataset_name: Name of the dataset being evaluated. Defaults to "Test".
        :type dataset_name: str
        :return: A dictionary containing accuracy, AUC-ROC score (if computed), predictions, and probabilities.
        :rtype: Dict
        """

        print(f"\nEvaluating on {dataset_name} set")

        # Predicciones
        y_pred = self.predict(code_pairs)
        y_proba = self.predict_proba(code_pairs)[:, 1]

        # Métricas
        accuracy = accuracy_score(labels, y_pred)

        try:
            auc = roc_auc_score(labels, y_proba)
        except:
            auc = None

        # Reporte de clasificación
        print(f"\n{dataset_name} Classification Report:")
        print("=" * 60)
        print(classification_report(labels, y_pred,
                                    target_names=['No Plagio', 'Plagio'],
                                    digits=3))

        if auc:
            print(f"AUC-ROC Score: {auc:.3f}")

        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def analyze_feature_importance(self, top_n: int = 20):
        """
        Analyze the importance of features based on the model's coefficients and display them
        in a sorted order. The function also provides statistics on feature types and allows
        visualizing the most important features up to a specified number.

        :param top_n: The number of most important features to display. Defaults to 20.
        :type top_n: int

        :return: A pandas DataFrame containing feature importance information with columns:
                 'feature', 'coefficient', and 'abs_importance'. The result is sorted by
                 absolute importance in descending order.
        :rtype: pandas.DataFrame
        """

        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Obtener coeficientes del modelo
        coefficients = self.model.coef_[0]

        # Crear DataFrame con importancia
        if hasattr(self.feature_selector, 'scores_'):
            selected_features = self.feature_selector.get_support()
            selected_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
        else:
            selected_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': selected_names,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        }).sort_values('abs_importance', ascending=False)

        # Mostrar características más importantes
        print(f"\nTop {top_n} most important features:")
        print("=" * 80)

        top_features = importance_df.head(top_n)
        for _, row in top_features.iterrows():
            feature_type = "ASTCC" if row['feature'].startswith('astcc_') else "TF-IDF"
            print(f"{row['feature']:<55} {feature_type:<8} {row['coefficient']:>8.3f}")

        # Estadísticas por tipo
        astcc_features = importance_df[importance_df['feature'].str.startswith('astcc_')]
        tfidf_features = importance_df[importance_df['feature'].str.startswith('tfidf_')]

        print(f"\nFeature type statistics:")
        print(f"ASTCC features: {len(astcc_features)} (avg importance: {astcc_features['abs_importance'].mean():.3f})")
        if len(tfidf_features) > 0:
            print(f"TF-IDF features: {len(tfidf_features)} (avg importance: {tfidf_features['abs_importance'].mean():.3f})")

        return importance_df

    def save_model(self, filepath: str):
        """
        Saves the trained model to the specified file path. The method first ensures
        that the model has been trained. If not, it raises an error. The specified
        file path's directory is created if it does not exist. The model is then
        serialized and written to a binary file at the provided file path. A success
        message is printed upon completion.

        :param filepath: The path where the trained model will be saved,
            including the filename.
        :type filepath: str
        :return: None
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to: {filepath}")


def load_code_pairs_from_csv(base_path: str, csv_path: str, split: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Loads code pairs and their associated labels from a specified CSV file. The function processes
    datasets 'ir_plag' and 'conplag' from the CSV and retrieves the respective code files based
    on the dataset type and folder/file structure. It ensures the files exist before attempting
    to read them. Successfully read code pairs are returned along with their labels.

    :param base_path: The root directory containing datasets.
    :type base_path: str
    :param csv_path: The file path to the input CSV containing dataset references.
    :type csv_path: str
    :param split: Subdirectory name (e.g., 'train', 'test', etc.) containing dataset splits.
    :type split: str
    :return: A tuple containing a list of code pairs (as tuples of strings) and a list of
        corresponding labels (as integers).
    :rtype: Tuple[List[Tuple[str, str]], List[int]]
    """

    df = pd.read_csv(csv_path)
    df = df[df['source_dataset'].isin(['ir_plag', 'conplag'])]

    code_pairs = []
    labels = []

    for _, row in df.iterrows():
        dataset = row['source_dataset']
        label = row['label']

        if dataset == 'ir_plag':
            folder = row['folder_name']
            folder_path = os.path.join(base_path, split, folder)
            path1 = os.path.join(folder_path, 'original.java')
            path2 = os.path.join(folder_path, 'compared.java')

        elif dataset == 'conplag':
            file1 = row['file1']
            file2 = row['file2']
            file1_base = os.path.splitext(file1)[0]
            file2_base = os.path.splitext(file2)[0]

            folder1 = os.path.join(base_path, split, f"{file1_base}_{file2_base}")
            folder2 = os.path.join(base_path, split, f"{file2_base}_{file1_base}")

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

        if os.path.exists(path1) and os.path.exists(path2):
            try:
                with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
                    code1 = f.read()
                with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
                    code2 = f.read()

                code_pairs.append((code1, code2))
                labels.append(label)

            except Exception as e:
                print(f"Error reading {path1}, {path2}: {e}")
                continue

    return code_pairs, labels


def train_and_evaluate_models():
    """
    Trains and evaluates three plagiarism detection models (TF-IDF Only, ASTCC Only, and Hybrid).
    For each model, training is performed, test and validation evaluations are conducted, and trained
    models are saved. Generates feature importance analysis and visualizations when tools are available.

    The models combine Advanced Syntax-Tree Coupled Comparison (ASTCC) and Term Frequency-Inverse
    Document Frequency (TF-IDF) features for identifying similarities in code pairs. This function
    also prepares final performance summaries and reports.

    :return:
        A tuple containing the dictionary of trained models, test evaluation results,
        and validation evaluation results.
    :rtype:
        tuple[dict[str, HybridPlagiarismClassifier], dict[str, dict], dict[str, dict]]
    """

    HYBRID_ASTCC_WEIGHT = 0.8
    HYBRID_TFIDF_WEIGHT = 0.2
    BASE_PATH = "data/splits"

    # Crear directorios
    os.makedirs("images", exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETO CON ASTCC")
    print("="*70)
    print(f"Configuración Híbrida: ASTCC={HYBRID_ASTCC_WEIGHT}, TF-IDF={HYBRID_TFIDF_WEIGHT}")

    # 1. Cargar datos
    print("\nCargando datasets...")
    train_pairs, train_labels = load_code_pairs_from_csv(BASE_PATH, os.path.join(BASE_PATH, "train.csv"), "train")
    val_pairs, val_labels = load_code_pairs_from_csv(BASE_PATH, os.path.join(BASE_PATH, "validation.csv"), "validation")
    test_pairs, test_labels = load_code_pairs_from_csv(BASE_PATH, os.path.join(BASE_PATH, "test.csv"), "test")

    print(f"Training: {len(train_pairs)} pairs")
    print(f"Validation: {len(val_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    # Cargar herramientas de visualización
    try:
        from visualization_tools import PlagiarismVisualizationTools
        viz_tools = PlagiarismVisualizationTools()
        print("Herramientas de visualización cargadas")
    except ImportError:
        viz_tools = None
        print("⚠Sin herramientas de visualización")

    models = {}
    test_results = {}
    val_results = {}

    print("\n" + "-"*50)
    print("MODELO 1/3: TF-IDF Only")
    print("-"*50)

    tfidf_classifier = HybridPlagiarismClassifier(use_tfidf=True, use_astcc=False)
    tfidf_classifier.train(train_pairs, train_labels)

    test_results['TF-IDF Only'] = tfidf_classifier.evaluate(test_pairs, test_labels, "TF-IDF Test")

    val_results['TF-IDF Only'] = tfidf_classifier.evaluate(val_pairs, val_labels, "TF-IDF Validation")

    # Guardar modelo
    tfidf_classifier.save_model("models/tfidf_only_model.pkl")
    models['tfidf'] = tfidf_classifier

    # 3. Entrenar y evaluar modelo ASTCC
    print("\n" + "-"*50)
    print("MODELO 2/3: ASTCC Only")
    print("-"*50)

    astcc_classifier = HybridPlagiarismClassifier(use_tfidf=False, use_astcc=True)
    astcc_classifier.train(train_pairs, train_labels)

    # Evaluar
    test_results['ASTCC Only'] = astcc_classifier.evaluate(test_pairs, test_labels, "ASTCC Test")
    val_results['ASTCC Only'] = astcc_classifier.evaluate(val_pairs, val_labels, "ASTCC Validation")

    # Guardar modelo
    astcc_classifier.save_model("models/astcc_only_model.pkl")
    models['astcc'] = astcc_classifier

    # 4. Entrenar y evaluar modelo híbrido
    print("\n" + "-"*50)
    print(f"MODELO 3/3: Hybrid (ASTCC:{HYBRID_ASTCC_WEIGHT}, TF-IDF:{HYBRID_TFIDF_WEIGHT})")
    print("-"*50)

    hybrid_classifier = HybridPlagiarismClassifier(
        use_tfidf=True,
        use_astcc=True,
        astcc_weight=HYBRID_ASTCC_WEIGHT,
        tfidf_weight=HYBRID_TFIDF_WEIGHT
    )
    hybrid_classifier.train(train_pairs, train_labels)

    # Evaluar
    test_results['Hybrid'] = hybrid_classifier.evaluate(test_pairs, test_labels, "Hybrid Test")
    val_results['Hybrid'] = hybrid_classifier.evaluate(val_pairs, val_labels, "Hybrid Validation")

    # Guardar modelo
    hybrid_classifier.save_model("models/hybrid_model.pkl")
    models['hybrid'] = hybrid_classifier

    # 5. Análisis de características importantes del modelo híbrido
    print("\n" + "-"*50)
    print("🔍 ANÁLISIS DE CARACTERÍSTICAS")
    print("-"*50)
    feature_importance = hybrid_classifier.analyze_feature_importance(top_n=30)

    # 6. Generar visualizaciones si están disponibles
    if viz_tools:
        print("\n" + "-"*50)
        print("🎨 GENERANDO VISUALIZACIONES")
        print("-"*50)

        # Preparar datos para visualización
        viz_data = {}
        for model_name in ['TF-IDF Only', 'ASTCC Only', 'Hybrid']:
            viz_data[model_name] = {
                'y_true': test_labels,
                'y_pred': test_results[model_name]['predictions'],
                'y_proba': test_results[model_name]['probabilities']
            }

        # Generar reporte visual
        summary_df = viz_tools.create_comprehensive_comparison_report(
            viz_data,
            feature_importance=feature_importance,
            dataset_name="Test"
        )

        summary_df.to_csv('csv/improvement/complete_analysis_summary.csv')

    # 7. Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN FINAL")
    print("="*70)

    print(f"\n{'Modelo':<15} {'Test Acc':<10} {'Test AUC':<10} {'Val Acc':<10} {'Val AUC':<10}")
    print("-" * 60)

    for model_name in ['TF-IDF Only', 'ASTCC Only', 'Hybrid']:
        test_acc = test_results[model_name]['accuracy']
        test_auc = test_results[model_name]['auc']
        val_acc = val_results[model_name]['accuracy']
        val_auc = val_results[model_name]['auc']

        print(f"{model_name:<15} {test_acc:<10.4f} {test_auc:<10.4f} {val_acc:<10.4f} {val_auc:<10.4f}")

    # Encontrar el mejor modelo
    best_model = max(test_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 MEJOR MODELO (por accuracy en test): {best_model[0]}")
    print(f"Test Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"Test AUC: {best_model[1]['auc']:.4f}")

    print("\nARCHIVOS GENERADOS:")
    print("models/tfidf_only_model.pkl")
    print("models/astcc_only_model.pkl")
    print("models/hybrid_model.pkl")

    if viz_tools:
        print("images/comparison_*.png")
        print("csv/improvement/complete_analysis_summary.csv")

    print(f"\nANÁLISIS COMPLETADO!")

    return models, test_results, val_results


def load_and_test_saved_models():
    """
    Función para cargar y probar los modelos guardados
    """
    print("\nCargando modelos guardados...")

    models = {}

    # Cargar modelos
    with open("models/tfidf_only_model.pkl", 'rb') as f:
        models['tfidf'] = pickle.load(f)
        print("Modelo TF-IDF cargado")

    with open("models/astcc_only_model.pkl", 'rb') as f:
        models['astcc'] = pickle.load(f)
        print("Modelo ASTCC cargado")

    with open("models/hybrid_model.pkl", 'rb') as f:
        models['hybrid'] = pickle.load(f)
        print("Modelo Híbrido cargado")

    # Cargar algunos datos de prueba
    BASE_PATH = "data/splits"
    test_pairs, test_labels = load_code_pairs_from_csv(
        BASE_PATH,
        os.path.join(BASE_PATH, "test.csv"),
        "test"
    )

    sample_pair = [test_pairs[0]]
    sample_label = test_labels[0]

    for name, model in models.items():
        pred = model.predict(sample_pair)[0]
        proba = model.predict_proba(sample_pair)[0, 1]
        print(f"{name}: Predicción={pred}, Probabilidad={proba:.3f} (Real={sample_label})")

    return models


if __name__ == "__main__":
    train_and_evaluate_models()

