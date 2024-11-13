import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from scikeras.wrappers import KerasRegressor
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from difflib import SequenceMatcher
from collections import Counter
import string
import nltk
import warnings
from gensim.models import Word2Vec

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
class SemanticSimilarityAnalyzer:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = string.punctuation
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.word2vec_model = None

    def preprocess_text(self, text):
        """Preprocess text with improved handling of special characters and punctuation"""
        text = text.replace("``", '"').replace("''", '"')
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        lemmatized = []
        custom_stops = self.stop_words - {'not', 'no', "don't", "doesn't", "didn't", "won't", "wouldn't"}
        
        for word, tag in tagged:
            if word not in custom_stops or word in {'?', '.', '!'}:
                if tag.startswith('N'):
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos='n'))
                elif tag.startswith('V'):
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos='v'))
                elif tag.startswith('R'):
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos='r'))
                elif tag.startswith('J'):
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos='a'))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(word))
        
        return lemmatized

    def extract_features(self, sent1, sent2):
        """
        Extract multiple similarity features between two sentences
        with proper handling of empty sentences
        """
        # Handle empty sentences
        if not sent1 and not sent2:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Length-based features
        if not sent1 or not sent2:
            len_diff = 1.0  # Maximum difference when one sentence is empty
        else:
            len_diff = abs(len(sent1) - len(sent2)) / max(len(sent1), len(sent2))
        
        # Word overlap features
        words1 = set(sent1)
        words2 = set(sent2)
        if not words1 and not words2:
            jaccard = 0.0
        else:
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Character-based similarity
        str1 = ' '.join(sent1) if sent1 else ''
        str2 = ' '.join(sent2) if sent2 else ''
        char_sim = SequenceMatcher(None, str1, str2).ratio()
        
        # Word frequency similarity
        freq1 = Counter(sent1)
        freq2 = Counter(sent2)
        freq_union = sum((freq1 | freq2).values())
        if freq_union > 0:
            freq_sim = sum((freq1 & freq2).values()) / freq_union
        else:
            freq_sim = 0.0
        
        return np.array([len_diff, jaccard, char_sim, freq_sim])

    def train_word2vec(self, sentences1, sentences2):
        """
        Train Word2Vec model on combined sentences
        """
        combined_sentences = sentences1 + sentences2
        self.word2vec_model = Word2Vec(sentences=combined_sentences, 
                                     vector_size=100, 
                                     window=5, 
                                     min_count=1, 
                                     workers=4)

    def get_sentence_embedding(self, sentence):
        """
        Get sentence embedding using Word2Vec
        """
        words = [word for word in sentence if word in self.word2vec_model.wv]
        if not words:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(self.word2vec_model.wv[words], axis=0)

    def get_transformer_embeddings(self, original_sentences):
        """
        Get sentence embeddings using transformer model
        """
        return self.sentence_transformer.encode(original_sentences)

    def optimize_svr(self, X, y):
        """
        Optimize SVR hyperparameters using GridSearchCV
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.1, 0.2]
        }
        
        svr = SVR()
        grid_search = GridSearchCV(
            svr, param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        print("\nBest parameters:", grid_search.best_params_)
        return grid_search.best_estimator_

    def train_ensemble(self, X_train, y_train):
        """
        Train an ensemble of models
        """
        models = {
            'svr': SVR(kernel='rbf', C=1.0),
            'rf': RandomForestRegressor(n_estimators=100),
            'ridge': Ridge(alpha=1.0)
        }
        
        trained_models = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models

    def predict_ensemble(self, models, X_test):
        """
        Make predictions using ensemble of models
        """
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_test)
        
        # Combine predictions using weighted average
        weights = {'svr': 0.4, 'rf': 0.4, 'ridge': 0.2}
        ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)
        
        return ensemble_pred, predictions

    def analyze_errors(self, y_true, y_pred, original_sentences1, original_sentences2):
        """
        Analyze and print the worst predictions
        """
        errors = np.abs(y_true - y_pred)
        error_indices = np.argsort(errors)[::-1]
        
        print("\nTop 5 worst predictions:")
        for idx in error_indices[:5]:
            print(f"\nSentence 1: {original_sentences1[idx]}")
            print(f"Sentence 2: {original_sentences2[idx]}")
            print(f"True similarity: {y_true[idx]:.2f}")
            print(f"Predicted similarity: {y_pred[idx]:.2f}")
            print(f"Error: {errors[idx]:.2f}")

    def evaluate_predictions(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate predictions using multiple metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        print(f"\n{model_name} Evaluation Results:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")

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
    analyzer = SemanticSimilarityAnalyzer()
    data_path = r'C:\AllMyCodes\IR\semantic_ana\Semantic-Analysis\data\dataset.txt'
    
    # Load and preprocess data
    original_sentences1, original_sentences2, preprocessed_sentences1, preprocessed_sentences2, labels = [], [], [], [], []
    
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            sent1, sent2, label = line.strip().split('\t')
            label = float(label)
            original_sentences1.append(sent1)
            original_sentences2.append(sent2)
            preprocessed_sentences1.append(analyzer.preprocess_text(sent1))
            preprocessed_sentences2.append(analyzer.preprocess_text(sent2))
            labels.append(label)
    
    labels = np.array(labels)
    
    # Generate embeddings and features
    transformer_embeddings1 = analyzer.get_transformer_embeddings(original_sentences1)
    transformer_embeddings2 = analyzer.get_transformer_embeddings(original_sentences2)
    transformer_similarity = np.array([np.dot(transformer_embeddings1[i], transformer_embeddings2[i]) / 
                                       (np.linalg.norm(transformer_embeddings1[i]) * np.linalg.norm(transformer_embeddings2[i])) 
                                       for i in range(len(transformer_embeddings1))])

    additional_features = np.array([analyzer.extract_features(s1, s2) for s1, s2 in zip(preprocessed_sentences1, preprocessed_sentences2)])
    X = np.hstack((transformer_similarity.reshape(-1, 1), additional_features))
    y = labels
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimize and train neural network model
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

    # Optionally, combine with SVR predictions
    print("Training and evaluating SVR model...")
    svr_model = analyzer.optimize_svr(X_train, y_train)
    svr_predictions = svr_model.predict(X_test)
    analyzer.evaluate_predictions(y_test, svr_predictions, model_name="SVR")

    # Ensemble predictions
    ensemble_predictions = 0.5 * nn_predictions + 0.5 * svr_predictions
    analyzer.evaluate_predictions(y_test, ensemble_predictions, model_name="Ensemble Model")

    # Error Analysis
    analyzer.analyze_errors(y_test, ensemble_predictions, original_sentences1, original_sentences2)

if __name__ == "__main__":
    main()