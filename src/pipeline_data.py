import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_train_test_data(filepath):
    """
    Carrega o dataset, separa as features e o target (Response)
    e realiza o particionamento estratificado inicial (80/20).
    """
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['Response'])
    y = df['Response']
    
    # Particionamento Honesto: 80% treino/validação e 20% teste final intocado
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    return X_train_val, X_test, y_train_val, y_test, X.columns

def create_modeling_pipeline(model):
    """
    Cria um Pipeline encapsulando o pré-processamento (StandardScaler)
    e o estimador de Machine Learning. Garante que o scaler seja ajustado
    apenas dentro de cada dobra (fold) durante a validação cruzada.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    return pipeline