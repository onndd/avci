
import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib
import os

class HMMRegimeDetector:
    def __init__(self, n_components=3):
        self.n_components = n_components
        # GaussianHMM: Assumes emissions are Gaussian
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        self.regime_map = {} # Maps hidden state ID to human meaning (0: Dead, 1: Normal, 2: Volatile)
        
    def fit(self, values):
        """
        Fits HMM on sequence of outcomes (Multiplier Values).
        Uses simple reshaped values or log-returns. Here using Multiplier directly (or Log(Multiplier)).
        """
        # Log Transform is better for Multipliers
        X_log = np.log1p(values).reshape(-1, 1)
        
        self.model.fit(X_log)
        print(f"HMM Fitted with {self.n_components} states.")
        
        # Analyze States to Label Them
        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Smallest Mean = Dead/Cold (0)
        # Medium Mean = Normal (1)
        # Largest Mean = Volatile/Hot (2)
        
        self.regime_map = {
            sorted_indices[0]: "COLD (Dead)",
            sorted_indices[1]: "NORMAL",
            sorted_indices[2]: "HOT (Volatile)"
        }
        
        print("Regime Mapping:", self.regime_map)
        
    def predict_state(self, values):
        X_log = np.log1p(values).reshape(-1, 1)
        states = self.model.predict(X_log)
        return states
    
    def predict_current_state(self, values):
        # Predicts state for the LAST sequence
        states = self.predict_state(values)
        return states[-1], self.regime_map.get(states[-1], "Unknown")
        
    def save(self, path='models/avci_hmm.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'map': self.regime_map}, path)
        
    def load(self, path='models/avci_hmm.pkl'):
        data = joblib.load(path)
        self.model = data['model']
        self.regime_map = data['map']

def train_hmm(df):
    print("--- Training HMM (Commander) ---")
    values = df['value'].values
    # We can use a subset to fit for speed if needed
    hmm_detector = HMMRegimeDetector(n_components=3)
    hmm_detector.fit(values)
    
    hmm_detector.save()
    print("HMM Saved.")
    return hmm_detector
