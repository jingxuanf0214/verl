import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu

#nltk.download('punkt', quiet=True)
nltk.download('punkt',download_dir='/n/home03/fjxdaisy/nltk_data',quiet=True)

class TextSimilarity:
    def __init__(self, method='jaccard'):
        available_methods = ['jaccard', 'dice', 'tfidf_cosine', 'overlap', 'bleu', 'hamming']
        if method not in available_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {available_methods}")
        self.method = method

    def compute_score(self, solution_str, ground_truth):
        method_func = getattr(self, f'_{self.method}')
        return method_func(solution_str, ground_truth)

    def _tokenize(self, text):
        return set(nltk.word_tokenize(text.lower()))

    def _tokenize_list(self, text):
        return nltk.word_tokenize(text.lower())

    def _jaccard(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0

    def _dice(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) else 0.0

    def _overlap(self, text1, text2):
        set1, set2 = self._tokenize(text1), self._tokenize(text2)
        intersection = len(set1 & set2)
        return intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) else 0.0

    def _tfidf_cosine(self, text1, text2):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2]).toarray()
        similarity = 1 - cosine(vectors[0], vectors[1])
        return float(similarity) if not np.isnan(similarity) else 0.0
    
    def _bleu(self, text1, text2):
        # BLEU score expects reference first, then candidate
        reference = [self._tokenize_list(text2)]  # List of reference sentences
        candidate = self._tokenize_list(text1)    # Single candidate sentence
        try:
            score = sentence_bleu(reference, candidate)
            # BLEU score is already between 0 and 1
            return float(score)
        except Exception:
            return 0.0  # Return 0 if BLEU calculation fails

    def _hamming(self, text1, text2):
        # Convert strings to lists of characters
        s1 = list(text1.lower())
        s2 = list(text2.lower())
        
        # If strings have different lengths, pad the shorter one with spaces
        max_len = max(len(s1), len(s2))
        s1.extend([' '] * (max_len - len(s1)))
        s2.extend([' '] * (max_len - len(s2)))
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        
        # Normalize to [0,1] range where 1 means identical strings
        return 1.0 - (distance / max_len)


