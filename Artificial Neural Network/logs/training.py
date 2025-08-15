import os
import sys
import logging
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

sys.path.append(os.getcwd())
from logs.constant import Constant
from logs.preprocessing import LogPreprocessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(n_jobs=-1, contamination=0.05, verbose=1)
        self.one_svm = OneClassSVM(verbose=1, nu=0.05)

    def train_models(self, X: np.ndarray):
        """Entraîne Isolation Forest et One-Class SVM et sauvegarde les modèles."""
        logging.info("Entraînement des modèles...")
        self.isolation_forest.fit(X)
        self.one_svm.fit(X)

        joblib.dump(self.isolation_forest, Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        joblib.dump(self.one_svm, Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)
        logging.info("Modèles entraînés et sauvegardés avec succès.")

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Prédit les anomalies avec Isolation Forest et One-Class SVM."""
        logging.info("Chargement des modèles pour la prédiction...")
        self._check_model_files()

        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        logging.info("Prédiction en cours...")
        return {
            "isolation_forest": isolation_forest.predict(X),
            "one_svm": one_svm.predict(X)
        }


    def evaluate_model(self, X: np.ndarray):
        """Évalue les modèles en affichant les distributions des scores d'anomalies."""
        logging.info("Évaluation des modèles en cours...")
        scores = self.compute_anomaly_score(X)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        ax1.hist(scores["isolation_forest"], bins=20, color='blue', alpha=0.7, edgecolor="black")
        ax1.set_title("Histogramme Isolation Forest")
        ax1.set_ylabel("Fréquence")

        ax2.hist(scores["one_svm"], bins=20, color='red', alpha=0.7, edgecolor="black")
        ax2.set_title("Histogramme One Class SVM")
        ax2.set_xlabel("Score d'anomalies")
        ax2.set_ylabel("Fréquence")

        plt.tight_layout()
        plt.show()

        prediction = self.predict(X)
        outlier_isolation_forest_ratio = np.mean(prediction["isolation_forest"] == -1)
        outlier_one_svm_ratio = np.mean(prediction["one_svm"] == -1)

        print(f"Ratio d'outlier IsolationForest: {outlier_isolation_forest_ratio:.2%}")
        print(f"Ratio d'outlier One Class SVM: {outlier_one_svm_ratio:.2%}")
        logging.info("Évaluation terminée.")

    def compute_anomaly_score(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Calcule les scores d'anomalies des modèles."""
        logging.info("Chargement des modèles pour le calcul des scores...")
        self._check_model_files()

        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        logging.info("Calcul des scores d'anomalie...")
        return {
            "isolation_forest": isolation_forest.decision_function(X),
            "one_svm": one_svm.decision_function(X)
        }

    def _check_model_files(self):
        """Vérifie si les fichiers modèles existent avant de les charger."""
        if not os.path.exists(Constant.ISOLATION_FOREST_MODEL_FILE_NAME):
            raise FileNotFoundError(f"Le modèle Isolation Forest est introuvable: {Constant.ISOLATION_FOREST_MODEL_FILE_NAME}")
        if not os.path.exists(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME):
            raise FileNotFoundError(f"Le modèle One Class SVM est introuvable: {Constant.ONE_CLASS_SVM_MODEL_FILE_NAME}")

if __name__ == "__main__":
    logging.info("Chargement du dataset...")
    df = pd.read_csv(Constant.LOGS_DATASET_FILENAME)

    logging.info("Prétraitement des données...")
    preprocessor = LogPreprocessor()
    df_train, df_test = preprocessor.split_dataset(df)

    X_train, df_train_engineered = preprocessor.fit_transform(df_train)
    X_test, df_test_engineered = preprocessor.fit_transform(df_test)

    detector = AnomalyDetector()
    
    detector.train_models(X_train)

    detector.evaluate_model(X_test)
    
    # detector.plot_isolation_forest_boundary(X_test)


