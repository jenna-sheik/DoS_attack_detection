import pandas as pd

# Load the data, specifying tab separation
df = pd.read_csv("packets.csv", sep="\t", header=None, 
                 names=["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "frame_len", "protocol", "tcp_flags", "udp_len"])

# Convert data types
df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
df["frame_len"] = pd.to_numeric(df["frame_len"], errors='coerce')
df["src_port"] = df["src_port"].fillna(-1).astype(int)
df["dst_port"] = df["dst_port"].fillna(-1).astype(int)

# Create flow_id
df["flow_id"] = df["src_ip"] + "-" + df["dst_ip"] + "-" + df["src_port"].astype(str) + "-" + df["dst_port"].astype(str) + "-" + df["protocol"].astype(str)

# Compute flow statisticswhere tshark

flow_duration = df.groupby("flow_id")["timestamp"].agg(lambda x: x.max() - x.min()).reset_index()
flow_duration.columns = ["flow_id", "Flow_Duration"]
flow_bytes = df.groupby("flow_id")["frame_len"].sum().reset_index()
flow_bytes.columns = ["flow_id", "Total_Bytes"]
flow_packets = df.groupby("flow_id")["timestamp"].count().reset_index()
flow_packets.columns = ["flow_id", "Total_Packets"]

# Merge initial stats
flow_stats = flow_duration.merge(flow_bytes, on="flow_id").merge(flow_packets, on="flow_id")
flow_stats["Flow_Bytes/s"] = flow_stats["Total_Bytes"] / flow_stats["Flow_Duration"].replace(0, 1)
flow_stats["Flow_Packets/s"] = flow_stats["Total_Packets"] / flow_stats["Flow_Duration"].replace(0, 1)

# Forward and backward packet classification
fwd_pkts = df[df["tcp_flags"] != "0x10"]  # Exclude ACK packets
bwd_pkts = df[df["tcp_flags"] == "0x10"]  # Only ACK packets

# Forward packet statistics
fwd_pkt_stats = fwd_pkts.groupby("flow_id")["frame_len"].agg(["mean", "max", "min", "std"]).reset_index()
fwd_pkt_stats.columns = ["flow_id", "Fwd_Pkt_Len_Mean", "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min", "Fwd_Pkt_Len_Std"]

# Backward packet statistics
bwd_pkt_stats = bwd_pkts.groupby("flow_id")["frame_len"].agg(["mean", "max", "min", "std"]).reset_index()
bwd_pkt_stats.columns = ["flow_id", "Bwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Max", "Bwd_Pkt_Len_Min", "Bwd_Pkt_Len_Std"]

# Compute Inter-Arrival Time (IAT)
df["iat"] = df.groupby("flow_id")["timestamp"].diff()
flow_iat_stats = df.groupby("flow_id")["iat"].agg(["mean", "std", "max", "min"]).reset_index()
flow_iat_stats.columns = ["flow_id", "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min"]

# Count forward and backward packets
tot_fwd_pkts = fwd_pkts.groupby("flow_id").size().reset_index(name="Tot_Fwd_Pkts")
tot_bwd_pkts = bwd_pkts.groupby("flow_id").size().reset_index(name="Tot_Bwd_Pkts")

# Compute Forward and Backward IAT
fwd_pkts["fwd_iat"] = fwd_pkts.groupby("flow_id")["timestamp"].diff()
bwd_pkts["bwd_iat"] = bwd_pkts.groupby("flow_id")["timestamp"].diff()
fwd_iat_stats = fwd_pkts.groupby("flow_id")["fwd_iat"].mean().reset_index()
bwd_iat_stats = bwd_pkts.groupby("flow_id")["bwd_iat"].mean().reset_index()
fwd_iat_stats.columns = ["flow_id", "Fwd_IAT_Mean"]
bwd_iat_stats.columns = ["flow_id", "Bwd_IAT_Mean"]

# Merge all features
final_df = (flow_stats
            .merge(fwd_pkt_stats, on="flow_id", how="left")
            .merge(bwd_pkt_stats, on="flow_id", how="left")
            .merge(flow_iat_stats, on="flow_id", how="left")
            .merge(tot_fwd_pkts, on="flow_id", how="left")
            .merge(tot_bwd_pkts, on="flow_id", how="left")
            .merge(fwd_iat_stats, on="flow_id", how="left")
            .merge(bwd_iat_stats, on="flow_id", how="left"))

# Fill missing values
final_df.fillna(0, inplace=True)

# Select only required features
selected_features = [
    "Flow_Duration", "Flow_Bytes/s", "Flow_Packets/s",
    "Fwd_Pkt_Len_Mean", "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min", "Fwd_Pkt_Len_Std",
    "Bwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Max", "Bwd_Pkt_Len_Min", "Bwd_Pkt_Len_Std",
    "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "Fwd_IAT_Mean", "Bwd_IAT_Mean"
]

# Keep only the selected features
final_df = final_df[selected_features]

# Save the final features
final_df.to_csv("network_flow_features6.csv", index=False)
print("Feature extraction complete. Saved to network_flow_features6.csv")
