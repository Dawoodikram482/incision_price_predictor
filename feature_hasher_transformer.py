from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from collections import Counter

# Custom Transformer for Feature Hashing
class FeatureHasherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='dict')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        token_dicts = (Counter(tokens) for tokens in X.str.split())
        hashed = self.hasher.transform(token_dicts)
        return hashed.toarray()
