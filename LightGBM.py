import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data_path = "data/dataset.txt"
with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize stop words and punctuation for preprocessing
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation

# Prepare lists to hold tokenized sentences and labels
sentences1 = []
sentences2 = []
labels = []

# Preprocess sentences
for line in lines:
    # Split the line into parts (sentence1, sentence2, and similarity label)
    parts = line.strip().split('\t')
    if len(parts) == 3:
        sent1, sent2, label = parts
        
        # Tokenize, convert to lowercase, and remove stopwords/punctuation for each sentence
        sent1_tokens = [word.lower() for word in word_tokenize(sent1) if word.lower() not in stop_words and word not in punctuation]
        sent2_tokens = [word.lower() for word in word_tokenize(sent2) if word.lower() not in stop_words and word not in punctuation]
        
        # Append preprocessed sentences and label to their respective lists
        sentences1.append(sent1_tokens)
        sentences2.append(sent2_tokens)
        labels.append(float(label))  # Convert label to float for regression
    else:
        print(f"Skipping line due to incorrect format: {line}")

# Combine all sentences for Word2Vec training
combined_sentences = sentences1 + sentences2

# Train Word2Vec model to get word embeddings
word2vec_model = Word2Vec(sentences=combined_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Function to get sentence embedding by averaging word vectors
def get_sentence_embedding(sentence):
    # Get embeddings for words in the sentence that are in the Word2Vec vocabulary
    words = [word for word in sentence if word in word2vec_model.wv]
    # If no words have embeddings, return a zero vector
    if len(words) == 0:
        return np.zeros(word2vec_model.vector_size)
    # Otherwise, return the mean embedding for the sentence
    return np.mean(word2vec_model.wv[words], axis=0)

# Generate sentence embeddings for both sentences in each pair
embeddings1 = np.array([get_sentence_embedding(sent) for sent in sentences1])
embeddings2 = np.array([get_sentence_embedding(sent) for sent in sentences2])

# Compute cosine similarity between sentence embeddings for each pair
similarity_scores = np.array([ 
    np.dot(embeddings1[i], embeddings2[i]) / 
    (np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[i])) 
    if np.linalg.norm(embeddings1[i]) and np.linalg.norm(embeddings2[i]) else 0 
    for i in range(len(embeddings1))
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(similarity_scores.reshape(-1, 1), labels, test_size=0.2, random_state=42)

# Train LightGBM model
lgbm_model = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.05, max_depth=5)
lgbm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lgbm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)  # Pearson correlation coefficient
spearman_corr, _ = spearmanr(y_test, y_pred)  # Spearman correlation coefficient

# Print evaluation results
print("Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print(f"Pearson Correlation: {pearson_corr}")
print(f"Spearman Correlation: {spearman_corr}")
