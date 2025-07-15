import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

# Constants
DATA_PATH = 'C:/Users/Brian Muema/Desktop/Data/data/file_.csv'
MODEL_PATH = 'anemia_model.pkl'

# 1. Load dataset
def load_data(path):
    return pd.read_csv(path)

# 2. Preprocess data
def preprocess_data(df):
    # Standardize column names (strip spaces)
    df.columns = df.columns.str.strip()
    # Encode categorical variables
    df['Sex'] = df['Sex'].str.strip().map({'M': 0, 'F': 1})
    df['Anaemic'] = df['Anaemic'].str.strip().map({'No': 0, 'Yes': 1})
    # Drop missing values
    df = df.dropna()
    return df

# 3. Define features and target
def get_features_and_target(df, target_col='Anaemic'):
    feature_cols = ['Sex', '%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']
    X = df[feature_cols]
    y = df[target_col]
    return X, y

# 4. Train model
def train_model(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# 5. Evaluate model
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

# 6. Save model
def save_model(clf, path):
    dump(clf, path)
    print(f'Model saved to {path}')

# Main routine
def main():
    # Load data
    df = load_data(DATA_PATH)
    # Preprocess
    df = preprocess_data(df)
    # Features and target
    X, y = get_features_and_target(df)
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train
    clf = train_model(X_train, y_train)
    # Evaluate
    evaluate_model(clf, X_test, y_test)
    # Save
    save_model(clf, MODEL_PATH)

if __name__ == '__main__':
    main() 