import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from logs.constant import Constant

class LogPreprocessor:
    def __init__(self):
        self.preprocessor = None
    
    def parse_features(self, df) -> pd.DataFrame:
        df = df.copy()  # Éviter la modification directe
        df['status_code'] = df['status_code'].astype(str)
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')  # Gestion des erreurs
        df['hour_of_day'] = df['timestamps'].dt.hour
        df['day_of_week'] = df['timestamps'].dt.dayofweek
        df['month'] = df['timestamps'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Agrégation des statistiques par utilisateur
        user_stats = df.groupby('user_ip').agg({
            'status_code': [
                ("error_count", lambda x: ((x.str.startswith("5")) | (x.str.startswith("4"))).sum()),
                ("unique_error_type", lambda x: x[x.str.startswith(("5", "4"))].nunique()),
            ],
            'response_time': [
                ("avg_response_time", 'mean'),
                ("max_response_time", 'max')
            ]
        }).reset_index()

        # Correction des noms de colonnes multi-index
        user_stats.columns = ['user_ip', 'error_count', 'unique_error_type', 'avg_response_time', 'max_response_time']
        
        df = df.merge(user_stats, on='user_ip', how="left")

        # Correction des intervalles de bins
        df["response_time_category"] = pd.cut(
            df['response_time'], bins=[-1, 100, 300, float("inf")], 
            labels=['fast', 'normal', 'slow']
        )

        # Correction des erreurs dans la condition de détection des anomalies
        df["is_potential_anomalous"] = (
            (df['response_time'] > 200) |
            (df['status_code'].str.startswith("5")) |
            (df['status_code'].str.startswith("4"))
        ).astype(int)

        return df
    
    def build_preprocessor(self):
        numeric_features = [
            "response_time", "hour_of_day", "day_of_week", "month", "error_count", 
            "unique_error_type", "avg_response_time", "max_response_time"
        ]
        
        categorical_features = [
            "is_weekend", "method", "end_point", "is_potential_anomalous", "response_time_category"
        ]
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return self
        
    
    def fit_transform(self, df):
        df_parsed = self.parse_features(df)
        self.build_preprocessor()

        # Suppression des colonnes inutiles pour l'entraînement du modèle
        X = df_parsed.drop(columns=['timestamps', 'status_code', 'user_ip'], errors='ignore')
        
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        return X_preprocessed, X
    
    
    def split_dataset(self, df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
        """split dataset into two sets: Train and test
        """
        df = df.sample(n=len(df))
        train_set_size = Constant.TRAIN_SET_RATIO * len(df)
        df_train = df.iloc[:int(train_set_size)]
        df_test = df.iloc[int(train_set_size):]
        
        return df_train, df_test




if __name__ == "__main__":
    df = pd.read_csv("data/logs_dataset.csv")

    logPreprocessor = LogPreprocessor()
    
    X_preprocessed, X = logPreprocessor.fit_transform(df)

    print("X_preprocessed Done!")
    print("X_preprocessed shape:", X_preprocessed.shape)
    print("X shape:", X.shape)
    
    print("X_preprocessed", X_preprocessed)
    
    print("X sample:\n", X.head())

    X.to_csv("data/logs_preprocessed.csv", index=False)

