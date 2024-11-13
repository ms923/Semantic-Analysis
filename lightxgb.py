import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
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
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class EnhancedSemanticSimilarity:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) - {'not', 'no', 'don\'t'}
        self.punctuation = string.punctuation
        self.lemmatizer = WordNetLemmatizer()
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.word2vec_model = None

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower().replace("``", '"').replace("''", '"'))
        tagged = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in tagged 
            if word not in self.stop_words and word not in self.punctuation
        ]
        return lemmatized

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    def train_word2vec(self, sentences1, sentences2):
        combined_sentences = sentences1 + sentences2
        self.word2vec_model = Word2Vec(sentences=combined_sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_sentence_embedding(self, sentence):
        words = [word for word in sentence if word in self.word2vec_model.wv]
        if not words:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(self.word2vec_model.wv[words], axis=0)

    def get_transformer_embeddings(self, sentences):
        return self.sentence_transformer.encode(sentences)

    def extract_features(self, sent1, sent2):
        len_diff = abs(len(sent1) - len(sent2)) / max(len(sent1), len(sent2)) if sent1 and sent2 else 1.0
        jaccard = len(set(sent1).intersection(sent2)) / len(set(sent1).union(sent2)) if sent1 and sent2 else 0.0
        char_sim = SequenceMatcher(None, ' '.join(sent1), ' '.join(sent2)).ratio()
        freq_sim = sum((Counter(sent1) & Counter(sent2)).values()) / (sum((Counter(sent1) | Counter(sent2)).values()) or 1.0)
        return np.array([len_diff, jaccard, char_sim, freq_sim])

# Load and preprocess data
data_path = r'C:\AllMyCodes\IR\semantic_ana\Semantic-Analysis\data\dataset.txt'
with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize analyzer and process sentences
analyzer = EnhancedSemanticSimilarity()
sentences1, sentences2, labels = [], [], []
original_sentences1, original_sentences2 = [], []

for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 3:
        sent1, sent2, label = parts
        preprocessed_sent1 = analyzer.preprocess_text(sent1)
        preprocessed_sent2 = analyzer.preprocess_text(sent2)
        if preprocessed_sent1 and preprocessed_sent2:
            sentences1.append(preprocessed_sent1)
            sentences2.append(preprocessed_sent2)
            labels.append(float(label))
            original_sentences1.append(sent1)
            original_sentences2.append(sent2)

# Train Word2Vec model and generate embeddings
analyzer.train_word2vec(sentences1, sentences2)
word2vec_embeddings1 = np.array([analyzer.get_sentence_embedding(sent) for sent in sentences1])
word2vec_embeddings2 = np.array([analyzer.get_sentence_embedding(sent) for sent in sentences2])

# Calculate cosine similarity between Word2Vec embeddings
similarity_scores = np.array([
    np.dot(word2vec_embeddings1[i], word2vec_embeddings2[i]) / 
    (np.linalg.norm(word2vec_embeddings1[i]) * np.linalg.norm(word2vec_embeddings2[i]))
    if np.linalg.norm(word2vec_embeddings1[i]) and np.linalg.norm(word2vec_embeddings2[i]) else 0 
    for i in range(len(word2vec_embeddings1))
])

# Generate transformer embeddings
transformer_embeddings1 = analyzer.get_transformer_embeddings(original_sentences1)
transformer_embeddings2 = analyzer.get_transformer_embeddings(original_sentences2)
transformer_similarity = np.array([
    np.dot(transformer_embeddings1[i], transformer_embeddings2[i]) / 
    (np.linalg.norm(transformer_embeddings1[i]) * np.linalg.norm(transformer_embeddings2[i]))
    for i in range(len(transformer_embeddings1))
])

# Extract additional features and combine all features
additional_features = np.array([
    analyzer.extract_features(s1, s2) for s1, s2 in zip(sentences1, sentences2)
])
X = np.hstack([
    similarity_scores.reshape(-1, 1),
    transformer_similarity.reshape(-1, 1),
    additional_features
])
y = np.array(labels)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for LightGBM and XGBoost models using GridSearchCV
lgbm_model = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.05, max_depth=5)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 150, 200],
}

# Tune XGBoost
grid_search_xgb = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Tune LightGBM
grid_search_lgbm = GridSearchCV(lgbm_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search_lgbm.fit(X_train, y_train)
best_lgbm_model = grid_search_lgbm.best_estimator_

# Ensemble model using stacking
stacked_model = StackingRegressor(
    estimators=[('xgb', best_xgb_model), ('lgbm', best_lgbm_model)],
    final_estimator=LGBMRegressor()
)
stacked_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = stacked_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Pearson Correlation: {pearson_corr}')
print(f'Spearman Correlation: {spearman_corr}')
