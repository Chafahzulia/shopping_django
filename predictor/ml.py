# predictor/ml.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

def load_data(filename):
    month_map = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
                 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    
    df = pd.read_csv(filename)
    
    
    df['Weekend'] = df['Weekend'].astype(int)
    df['VisitorType'] = df['VisitorType'].apply(lambda x: 1 if x == 'Returning_Visitor' else 0)
    df['Month'] = df['Month'].map(month_map)
    df['Revenue'] = df['Revenue'].astype(int)
    
    evidence = df.drop('Revenue', axis=1)
    labels = df['Revenue']
    
    return evidence, labels

def train_model(evidence, labels):
    model = GaussianNB()
    model.fit(evidence, labels)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    return acc
