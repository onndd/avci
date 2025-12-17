
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AutoencoderDetector:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.threshold = 0.0
        
    def _build_model(self):
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Bottleneck (Latent Space)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded) # Linear because we reconstruct standard scaled features
        
        return Model(inputs=input_layer, outputs=output_layer)
        
    def fit(self, X):
        # Scale Data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Autoencoder
        # We want it to learn "Normal" patterns well.
        self.model.compile(optimizer='adam', loss='mse')
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=20,
            batch_size=64,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Determine Anomaly Threshold (e.g., 95th percentile of reconstruction error)
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        print(f"AE Trained. Anomaly Threshold (MSE): {self.threshold:.4f}")
        
    def predict_anomaly_score(self, X):
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        return mse
        
    def is_anomaly(self, X):
        scores = self.predict_anomaly_score(X)
        return (scores > self.threshold).astype(int)
        
    def save(self, path_prefix='models/avci_ae'):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        self.model.save(f"{path_prefix}_model.keras")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        with open(f"{path_prefix}_threshold.txt", "w") as f:
            f.write(str(self.threshold))
            
    def load(self, path_prefix='models/avci_ae'):
        self.model = load_model(f"{path_prefix}_model.keras")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        with open(f"{path_prefix}_threshold.txt", "r") as f:
            self.threshold = float(f.read())

def train_autoencoder(df):
    """
    Wrapper to train AE on 'Normal' data only (1.0x - 3.0x).
    """
    print("--- Training Autoencoder (Intelligence) ---")
    # Filter Normal Data
    # Identify 'Normal' as distinct from High X (>3.0).
    # Actually, AE should learn the majority. Majority is 90% of data.
    # So training on full dataset (unsupervised) is usually fine, 
    # OR training specifically on < 3.0 data to make High X 'weird'.
    
    mask_normal = (df['value'] < 5.0) # Train on "Boring" games
    df_normal = df[mask_normal]
    
    # Feature Selection (Numerical only)
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X_normal = df_normal[features]
    
    ae = AutoencoderDetector(input_dim=len(features))
    ae.fit(X_normal)
    
    ae.save()
    print("Autoencoder Saved.")
    return ae
