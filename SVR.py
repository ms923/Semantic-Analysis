import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from collections import Counter
import string
import nltk
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class SemanticSimilarityAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = string.punctuation
        self.lemmatizer = WordNetLemmatizer()
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.word2vec_model = None
        

    def preprocess_text(self, text):
        """
        Preprocess text with improved handling of special characters and punctuation
        """
        # Basic cleaning while preserving important punctuation
        text = text.replace("``", '"').replace("''", '"')
        
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # POS tag and lemmatize
        tagged = pos_tag(tokens)
        lemmatized = []
        
        # Custom stopwords that preserve important words
        custom_stops = set(stopwords.words('english')) - {'not', 'no', 'don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t'}
        
        for word, tag in tagged:
            # Keep words that aren't stopwords, unless they're important punctuation
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
        
        if not lemmatized:
            print(f"Warning: Empty result after preprocessing text: '{text}'")
        
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

def main():
    # Initialize analyzer
    analyzer = SemanticSimilarityAnalyzer()
    
    # Load dataset
    data_path = r'C:\AllMyCodes\IR\semantic_ana\Semantic-Analysis\data\dataset.txt'
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Prepare data
    original_sentences1 = []
    original_sentences2 = []
    preprocessed_sentences1 = []
    preprocessed_sentences2 = []
    labels = []
    
    # Process sentences
    print("Processing sentences...")
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Split by tab while handling possible empty fields
                    parts = line.strip().split('\t')
                    
                    if len(parts) != 3:
                        print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
                        continue
                        
                    sent1, sent2, label = parts
                    
                    # Convert label to float with error handling
                    try:
                        label_value = float(label)
                    except ValueError:
                        print(f"Warning: Invalid label value at line {line_num}: {label}")
                        continue
                    
                    # Store original sentences
                    original_sentences1.append(sent1)
                    original_sentences2.append(sent2)
                    
                    # Preprocess sentences
                    proc_sent1 = analyzer.preprocess_text(sent1)
                    proc_sent2 = analyzer.preprocess_text(sent2)
                    
                    # Only add if preprocessing produced non-empty results
                    if proc_sent1 and proc_sent2:
                        preprocessed_sentences1.append(proc_sent1)
                        preprocessed_sentences2.append(proc_sent2)
                        labels.append(label_value)s
                    else:
                        print(f"Warning: Skipping line {line_num} due to empty preprocessing result")
                        print(f"Original sentences: '{sent1}' | '{sent2}'")
                
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find dataset file at {data_path}")
        return
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
        return
    
    if not labels:
        print("Error: No valid data was loaded from the dataset")
        return
        
    print(f"Successfully loaded {len(labels)} valid sentence pairs")
    
    # Convert to numpy array
    labels = np.array(labels)
    
    print("Training Word2Vec model...")
    # Train Word2Vec model
    analyzer.train_word2vec(preprocessed_sentences1, preprocessed_sentences2)
    
    print("Generating embeddings...")
    # Generate different types of embeddings and features
    word2vec_embeddings1 = np.array([analyzer.get_sentence_embedding(sent) for sent in preprocessed_sentences1])
    word2vec_embeddings2 = np.array([analyzer.get_sentence_embedding(sent) for sent in preprocessed_sentences2])
    
    # Calculate cosine similarity between Word2Vec embeddings
    similarity_scores = np.array([
        np.dot(word2vec_embeddings1[i], word2vec_embeddings2[i]) / 
        (np.linalg.norm(word2vec_embeddings1[i]) * np.linalg.norm(word2vec_embeddings2[i])) 
        if np.linalg.norm(word2vec_embeddings1[i]) and np.linalg.norm(word2vec_embeddings2[i]) else 0 
        for i in range(len(word2vec_embeddings1))
    ])
    
    # Generate transformer embeddings
    print("Generating transformer embeddings...")
    transformer_embeddings1 = analyzer.get_transformer_embeddings(original_sentences1)
    transformer_embeddings2 = analyzer.get_transformer_embeddings(original_sentences2)
    
    # Calculate cosine similarity between transformer embeddings
    transformer_similarity = np.array([
        np.dot(transformer_embeddings1[i], transformer_embeddings2[i]) / 
        (np.linalg.norm(transformer_embeddings1[i]) * np.linalg.norm(transformer_embeddings2[i]))
        for i in range(len(transformer_embeddings1))
    ])
    
    # Extract additional features
    print("Extracting additional features...")
    additional_features = np.array([
        analyzer.extract_features(s1, s2) 
        for s1, s2 in zip(preprocessed_sentences1, preprocessed_sentences2)
    ])
    
    # Combine all features
    X = np.hstack([
        similarity_scores.reshape(-1, 1),
        transformer_similarity.reshape(-1, 1),
        additional_features
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Train and evaluate optimized SVR
    print("\nOptimizing SVR model...")
    best_svr = analyzer.optimize_svr(X_train, y_train)
    svr_predictions = best_svr.predict(X_test)
    analyzer.evaluate_predictions(y_test, svr_predictions, "Optimized SVR")
    
    # Train and evaluate ensemble
    print("\nTraining ensemble models...")
    ensemble_models = analyzer.train_ensemble(X_train, y_train)
    ensemble_predictions, individual_predictions = analyzer.predict_ensemble(ensemble_models, X_test)
    analyzer.evaluate_predictions(y_test, ensemble_predictions, "Ensemble")
    
    # Analyze errors
    print("\nAnalyzing errors...")
    analyzer.analyze_errors(y_test, ensemble_predictions, original_sentences1, original_sentences2)

if __name__ == "__main__":
    main()