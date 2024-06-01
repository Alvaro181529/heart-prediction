import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

# Entrenar y guardar el modelo
def train_and_save_model():
    dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    X = dataset[features]
    y = dataset['DEATH_EVENT']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(sampling_strategy='minority')
    X_res, y_res = smote.fit_resample(pd.DataFrame(X_scaled, columns=features), y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    tree = DecisionTreeClassifier(max_depth=6, criterion='entropy')
    tree.fit(X_train, y_train)
    dump(tree, 'modelo_heart_failure_tree.joblib')
    dump(scaler, 'scaler.joblib')  # Guardar el scaler
    return tree

# Cargar el modelo y hacer predicciones
def predict(data):
    model = load('modelo_heart_failure_tree.joblib')
    scaler = load('scaler.joblib')
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction
