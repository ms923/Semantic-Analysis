import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer


# Preprocessing utility function for text data
class TextAnalyzer:
    @staticmethod
    def preprocess_text(text):
        # Basic preprocessing: remove punctuation, convert to lowercase, etc.
        # You can customize this method to add more preprocessing steps
        text = text.lower().strip()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text

    @staticmethod
    def analyze_errors(y_true, y_pred, original_sent1, original_sent2):
        # Print out some sample mismatches for error analysis
        errors = abs(y_true - y_pred)
        error_threshold = np.percentile(errors, 90)  # Top 10% largest errors
        print("\nSamples with largest errors:")
        
        for i, error in enumerate(errors):
            if error >= error_threshold:
                print(f"Sentence 1: {original_sent1[i]}")
                print(f"Sentence 2: {original_sent2[i]}")
                print(f"True Label: {y_true[i]}, Predicted: {y_pred[i]}, Error: {error}")
                print("----")

class SemanticSimilarityNN:
    def __init__(self, input_dim, units1=128, units2=64, dropout1=0.2, dropout2=0.2, lr=0.001):
        self.input_dim = input_dim
        self.units1 = units1
        self.units2 = units2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(self.units1, activation='relu'))
        model.add(Dropout(self.dropout1))
        model.add(Dense(self.units2, activation='relu'))
        model.add(Dropout(self.dropout2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.lr))
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[early_stopping])

    def predict(self, X):
        return self.model.predict(X).flatten()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        print("Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        
        return y_pred

def optimize_nn(X, y):
    def create_model(units1=128, units2=64, dropout1=0.2, dropout2=0.2, lr=0.001):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(units1, activation='relu'))
        model.add(Dropout(dropout1))
        model.add(Dense(units2, activation='relu'))
        model.add(Dropout(dropout2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
        return model

    model = KerasRegressor(model=create_model, verbose=1)

    param_grid = {
        'model__units1': [64, 128, 256],
        'model__units2': [32, 64, 128],
        'model__dropout1': [0.1, 0.2, 0.3],
        'model__dropout2': [0.1, 0.2, 0.3],
        'model__lr': [0.001, 0.0001]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search

def embed_sentences(sentences):
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Generate sentence embeddings
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings


def main():
    data_path = r'C:\AllMyCodes\IR\semantic_ana\Semantic-Analysis\data\dataset.txt'
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    original_sentences1 = []
    original_sentences2 = []
    preprocessed_sentences1 = []
    preprocessed_sentences2 = []
    labels = []

    print("Processing sentences...")
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue

                    sent1, sent2, label = parts
                    try:
                        label_value = float(label)
                    except ValueError:
                        continue
                    
                    original_sentences1.append(sent1)
                    original_sentences2.append(sent2)
                    
                    proc_sent1 = TextAnalyzer.preprocess_text(sent1)
                    proc_sent2 = TextAnalyzer.preprocess_text(sent2)
                    
                    if proc_sent1 and proc_sent2:
                        preprocessed_sentences1.append(proc_sent1)
                        preprocessed_sentences2.append(proc_sent2)
                        labels.append(label_value)
                
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue

    except FileNotFoundError:
        print(f"Error: Could not find dataset file at {data_path}")
        return

    labels = np.array(labels)
    print(f"Successfully loaded {len(labels)} valid sentence pairs")

    # Convert sentences to embeddings
    sentence_embeddings1 = embed_sentences(preprocessed_sentences1)
    sentence_embeddings2 = embed_sentences(preprocessed_sentences2)

    # Concatenate embeddings for sentence pairs to form input vectors
    X = np.hstack([sentence_embeddings1, sentence_embeddings2])

    if X.shape[0] != len(labels):
        print("Error: Mismatch in number of samples between X and labels.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    print("Optimizing neural network model...")
    grid_search = optimize_nn(X_train, y_train)

    print("Training and evaluating optimized neural network model...")
    best_params = grid_search.best_params_
    nn_analyzer = SemanticSimilarityNN(
        input_dim=X_train.shape[1],
        units1=best_params['model__units1'],
        units2=best_params['model__units2'],
        dropout1=best_params['model__dropout1'],
        dropout2=best_params['model__dropout2'],
        lr=best_params['model__lr']
    )
    nn_analyzer.train_model(X_train, y_train, X_test, y_test)
    nn_predictions = nn_analyzer.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()