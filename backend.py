from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import threading
import time
import os
from io import StringIO

app = FastAPI()

# < CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginCredentials(BaseModel):
    username: str
    password: str

USER_DB = {"pranav": "appuranked"}

model = tf.keras.models.load_model("dos_attack_lstm15.h5")
scaler = joblib.load("scaler15.pkl")

output_dir = os.path.join(os.getcwd(), "data")
os.makedirs(output_dir, exist_ok=True)
TSHARK_OUTPUT = os.path.join(output_dir, "captured.csv")

selected_features = [
    "Flow Duration", "Flow Byts/s", "Flow Pkts/s",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Std",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Std",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "Fwd IAT Mean", "Bwd IAT Mean"
]

packet_counter = 0
current_packets = []
latest_prediction = None
progress_percentage = 0
last_read_position = 0

def start_tshark_capture():
    try:
        if os.path.exists(TSHARK_OUTPUT):
            with open(TSHARK_OUTPUT, "w") as f:
                f.write("")

        print("[+] Starting tshark...")

        interface = r"\Device\NPF_Loopback"  

        subprocess.Popen(
            [
                "tshark", "-i", interface,
                "-T", "fields",
                "-e", "frame.time_relative", "-e", "frame.len",
                "-e", "ip.src", "-e", "ip.dst",
                "-e", "tcp.srcport", "-e", "tcp.dstport",
                "-e", "udp.srcport", "-e", "udp.dstport",
                "-e", "frame.protocols",
                "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"
            ],
            stdout=open(TSHARK_OUTPUT, "w"),
            stderr=subprocess.DEVNULL
        )

    except Exception as e:
        print("[-] Tshark error:", e)

def extract_features(df):
    df.columns = [
        "frame.time_relative", "frame.len", "ip.src", "ip.dst",
        "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "frame.protocols"
    ]

    df = df.dropna(subset=["frame.time_relative", "frame.len", "ip.src", "ip.dst"])
    df["frame.time_relative"] = pd.to_numeric(df["frame.time_relative"], errors="coerce")
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")
    df = df.dropna(subset=["frame.time_relative", "frame.len"])
    df = df.sort_values(by="frame.time_relative").reset_index(drop=True)

    df["Flow Duration"] = df["frame.time_relative"].diff().fillna(0)
    df["Flow Byts/s"] = df["frame.len"] / df["Flow Duration"].replace(0, 1e-5)
    df["Flow Pkts/s"] = 1 / df["Flow Duration"].replace(0, 1e-5)

    window = 5
    fwd_src = df.iloc[0]["ip.src"]
    fwd_dst = df.iloc[0]["ip.dst"]
    df["direction"] = np.where((df["ip.src"] == fwd_src) & (df["ip.dst"] == fwd_dst), "Fwd", "Bwd")

    df["Fwd Pkt Len Mean"] = df[df["direction"] == "Fwd"]["frame.len"].rolling(window).mean().fillna(0)
    df["Fwd Pkt Len Max"] = df[df["direction"] == "Fwd"]["frame.len"].rolling(window).max().fillna(0)
    df["Fwd Pkt Len Min"] = df[df["direction"] == "Fwd"]["frame.len"].rolling(window).min().fillna(0)
    df["Fwd Pkt Len Std"] = df[df["direction"] == "Fwd"]["frame.len"].rolling(window).std().fillna(0)

    df["Bwd Pkt Len Mean"] = df[df["direction"] == "Bwd"]["frame.len"].rolling(window).mean().fillna(0)
    df["Bwd Pkt Len Max"] = df[df["direction"] == "Bwd"]["frame.len"].rolling(window).max().fillna(0)
    df["Bwd Pkt Len Min"] = df[df["direction"] == "Bwd"]["frame.len"].rolling(window).min().fillna(0)
    df["Bwd Pkt Len Std"] = df[df["direction"] == "Bwd"]["frame.len"].rolling(window).std().fillna(0)

    df["Flow IAT Mean"] = df["Flow Duration"].rolling(window).mean().fillna(0)
    df["Flow IAT Max"] = df["Flow Duration"].rolling(window).max().fillna(0)
    df["Flow IAT Min"] = df["Flow Duration"].rolling(window).min().fillna(0)
    df["Flow IAT Std"] = df["Flow Duration"].rolling(window).std().fillna(0)

    df["Fwd IAT Mean"] = df[df["direction"] == "Fwd"]["Flow Duration"].rolling(window).mean().fillna(0)
    df["Bwd IAT Mean"] = df[df["direction"] == "Bwd"]["Flow Duration"].rolling(window).mean().fillna(0)

    df["Tot Fwd Pkts"] = df["direction"].eq("Fwd").astype(int).cumsum()
    df["Tot Bwd Pkts"] = df["direction"].eq("Bwd").astype(int).cumsum()

    return df[selected_features].fillna(0)

PACKET_THRESHOLD = 15
READ_INTERVAL = 5

def process_packets():
    global packet_counter, current_packets, latest_prediction, progress_percentage, last_read_position

    while True:
        try:
            if not os.path.exists(TSHARK_OUTPUT):
                print("[-] Waiting for tshark file...")
                time.sleep(READ_INTERVAL)
                continue

            with open(TSHARK_OUTPUT, "r") as f:
                f.seek(last_read_position)
                new_lines = f.readlines()
                last_read_position = f.tell()

            if not new_lines:
                time.sleep(READ_INTERVAL)
                continue

            # Skip the last line assuming it might be in progress
            if len(new_lines) > 1:
                new_lines = new_lines[:-1]
            else:
                time.sleep(READ_INTERVAL)
                continue

            new_data = "\n".join([line.strip() for line in new_lines if line.strip()])
            df = pd.read_csv(StringIO(new_data), header=None, on_bad_lines="skip")

            if df.empty:
                time.sleep(READ_INTERVAL)
                continue

            packet_counter += len(df)
            current_packets.append(df)
            progress_percentage = min(int((packet_counter / PACKET_THRESHOLD) * 100), 100)

            if packet_counter >= PACKET_THRESHOLD:
                combined_df = pd.concat(current_packets, ignore_index=True)
                feature_df = extract_features(combined_df)

                if feature_df.empty:
                    time.sleep(READ_INTERVAL)
                    continue

                X_scaled = scaler.transform(feature_df)
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                prediction = model.predict(X_reshaped)
                latest_prediction = int(np.mean(prediction) > 0.6)

                print(f"[+] Prediction: {latest_prediction}")

                packet_counter = 0
                current_packets = []
                progress_percentage = 0

        except Exception as e:
            print(f"[-] Processing error: {e}")

        time.sleep(READ_INTERVAL)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=start_tshark_capture, daemon=True).start()
    threading.Thread(target=process_packets, daemon=True).start()

@app.post("/login")
async def login(credentials: LoginCredentials):
    if USER_DB.get(credentials.username) == credentials.password:
        return {"message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid username or password")

@app.get("/predict")
async def get_latest_prediction():
    if latest_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not ready")
    return {"prediction": "DoS Detected" if latest_prediction == 1 else "No DoS Detected"}

@app.get("/progress")
async def get_progress():
    return JSONResponse(content={"progress": progress_percentage})