import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your data from the file
data_path = r'C:\AllMyCodes\IR\semantic_ana\Semantic-Analysis\data\dataset.txt'

# Assuming your dataset is structured like: sentence1 \t sentence2 \t similarity_score
# Load the data
data = pd.read_csv(data_path, sep="\t", header=None, names=["sentence1", "sentence2", "true_similarity"])

# Initialize the pre-trained BERT model (Sentence-BERT or other)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to compute similarity between sentence embeddings
def compute_similarity(sent1, sent2):
    # Generate embeddings for both sentences
    embeddings1 = model.encode([sent1])
    embeddings2 = model.encode([sent2])
    
    # Compute cosine similarity (or use any other metric)
    cosine_similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
    return cosine_similarity[0][0]

# Predict similarities and calculate errors
predicted_similarities = []
true_similarities = data["true_similarity"].tolist()

for _, row in data.iterrows():
    pred_sim = compute_similarity(row["sentence1"], row["sentence2"])
    predicted_similarities.append(pred_sim)

# Evaluate the model's performance
mse = mean_squared_error(true_similarities, predicted_similarities)
pearson_corr, _ = pearsonr(true_similarities, predicted_similarities)
spearman_corr, _ = spearmanr(true_similarities, predicted_similarities)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Pearson Correlation: {pearson_corr}")
print(f"Spearman Correlation: {spearman_corr}")

# Show the top 5 worst predictions
errors = np.abs(np.array(predicted_similarities) - np.array(true_similarities))
worst_indices = np.argsort(errors)[-5:]

print("\nTop 5 worst predictions:")
for i in worst_indices:
    print(f"Sentence 1: {data.iloc[i]['sentence1']}")
    print(f"Sentence 2: {data.iloc[i]['sentence2']}")
    print(f"True similarity: {data.iloc[i]['true_similarity']}")
    print(f"Predicted similarity: {predicted_similarities[i]}")
    print(f"Error: {errors[i]}")
    print("-" * 60)
