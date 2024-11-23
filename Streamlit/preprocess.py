import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from difflib import SequenceMatcher
from collections import Counter
import string
import nltk

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def extract_features(sent1, sent2):
    """Extract similarity features between two preprocessed sentences."""
    len_diff = abs(len(sent1) - len(sent2)) / max(len(sent1), len(sent2)) if sent1 and sent2 else 1.0
    jaccard = len(set(sent1).intersection(sent2)) / len(set(sent1).union(sent2)) if sent1 and sent2 else 0.0
    char_sim = SequenceMatcher(None, ' '.join(sent1), ' '.join(sent2)).ratio()
    freq1, freq2 = Counter(sent1), Counter(sent2)
    freq_union = sum((freq1 | freq2).values())
    freq_sim = sum((freq1 & freq2).values()) / freq_union if freq_union > 0 else 0.0

    return np.array([len_diff, jaccard, char_sim, freq_sim])
