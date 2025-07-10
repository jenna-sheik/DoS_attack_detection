import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import classification_report

# ✅ Load the dataset
df = pd.read_csv("02-15-2018.csv", low_memory=False)

# ✅ Select relevant features
selected_features = [
    "Flow Duration", "Flow Byts/s", "Flow Pkts/s",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Std",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Std",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd IAT Mean", "Bwd IAT Mean"
]

# ✅ Keep only existing features
existing_features = [col for col in selected_features if col in df.columns]
df = df[existing_features + ["Label"]]

# ✅ Convert features to numeric
df[existing_features] = df[existing_features].apply(pd.to_numeric, errors='coerce')

# ✅ Replace inf and handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[existing_features] = df[existing_features].fillna(df[existing_features].median())  # Fill missing with median only for numeric columns

# ✅ Encode labels (Benign = 0, DoS attacks = 1)
df["Label"] = df["Label"].apply(lambda x: 1 if "DoS" in x else 0)

# ✅ Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(df[existing_features])
joblib.dump(scaler, "scaler15.pkl")

y = df["Label"].values

# ✅ Split data before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Reduce dataset before applying SMOTE (if needed, modify this)
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=100000, random_state=42, stratify=y_train)

# ✅ Create sequences for LSTM **before applying SMOTE**
sequence_length = 20  # Time steps

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# ✅ Reduce dataset size for SMOTE if necessary
X_train_seq = X_train_seq[:50000]  # Limit dataset size
y_train_seq = y_train_seq[:50000]

# ✅ Apply SMOTE after creating sequences (Reshape first)
X_train_2D = X_train_seq.reshape(X_train_seq.shape[0], -1)  # Flatten for SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Limit synthetic sample generation
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_2D, y_train_seq)

# ✅ Reshape back to 3D after SMOTE
X_train_resampled = X_train_resampled.reshape(-1, sequence_length, len(existing_features))

print("✅ Applied SMOTE. New class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# ✅ Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, len(existing_features))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train Model
history = model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(X_test_seq, y_test_seq)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Save Model
model.save("dos_attack_lstm15.h5")
print("✅ Model saved as 'dos_attack_lstm15.h5'")

# ✅ Make Predictions
y_pred = (model.predict(X_test_seq) > 0.5).astype("int32")

# ✅ Print Classification Report
print(classification_report(y_test_seq, y_pred, target_names=["Benign", "DoS Attack"]))
