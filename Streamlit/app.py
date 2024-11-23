import streamlit as st
import numpy as np
from gensim.models import Word2Vec
import joblib
from analyze_similarity import SemanticSimilarityAnalyzer

# Load models
@st.cache_resource
def load_models():
    word2vec_model = Word2Vec.load("models/word2vec_model")
    svr_model = joblib.load("models/svr_model.pkl")
    return word2vec_model, svr_model

# Streamlit app
def main():
    st.title("Semantic Similarity Analyzer")
    st.write("Enter two sentences to compute their similarity.")

    sentence1 = st.text_area("Sentence 1", height=100)
    sentence2 = st.text_area("Sentence 2", height=100)

    if st.button("Compute Similarity"):
        if sentence1.strip() and sentence2.strip():
            word2vec_model, svr_model = load_models()
            analyzer = SemanticSimilarityAnalyzer()
            analyzer.word2vec_model = word2vec_model

            # Preprocess and extract features
            preprocessed1 = analyzer.preprocess_text(sentence1)
            preprocessed2 = analyzer.preprocess_text(sentence2)
            word2vec_emb1 = analyzer.get_sentence_embedding(preprocessed1)
            word2vec_emb2 = analyzer.get_sentence_embedding(preprocessed2)
            word2vec_similarity = np.dot(word2vec_emb1, word2vec_emb2) / (
                np.linalg.norm(word2vec_emb1) * np.linalg.norm(word2vec_emb2)
            )
            transformer_emb1 = analyzer.get_transformer_embeddings([sentence1])[0]
            transformer_emb2 = analyzer.get_transformer_embeddings([sentence2])[0]
            transformer_similarity = np.dot(transformer_emb1, transformer_emb2) / (
                np.linalg.norm(transformer_emb1) * np.linalg.norm(transformer_emb2)
            )
            additional_features = analyzer.extract_features(preprocessed1, preprocessed2)

            X = np.hstack((
                np.array([word2vec_similarity]).reshape(1, -1),
                np.array([transformer_similarity]).reshape(1, -1),
                additional_features.reshape(1, -1)
            ))

            # Predict similarity
            similarity = svr_model.predict(X)[0]
            st.success(f"Predicted Similarity: {similarity:.2f}")
        else:
            st.error("Please provide valid sentences.")

if __name__ == "__main__":
    main()
