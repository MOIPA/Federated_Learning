import pandas as pd

# Load your dataset
dataset_path = "Federated_Learning-main\cyber_physic_dataset\Dataset.csv"  # Update this path to your actual dataset location

data = pd.read_csv(dataset_path, header=None)

segments = {
    "Benign_Cyber": (0, 9426),  # 从0开始以包括标题行
    "Benign_Physical": (9426, 13717),  # 从上一段的结束开始，保留标题行
    "DoS_Attack_Cyber": (13717, 25389),
    "DoS_Attack_Physical": (25389, 26363),
    "Replay_Attack_Cyber": (26363, 38370),
    "Replay_Attack_Physical": (38370, 39344),
    "Evil_Twin_Cyber": (39344, 45028),
    "Evil_Twin_Physical": (45028, 50502),
    "FDI_Cyber": (50502, 53976),
    "FDI_Physical": (53976, 54784),
}


# Segment the dataset and save each segment to a new CSV file
for segment_name, (start, end) in segments.items():
    # Adjust for zero-based indexing
    segment_data = data.iloc[start:end]

    # Define the path for the new CSV file
    # Replace 'your_directory_path' with the path where you want to save the files
    save_path = (
        f"Federated_Learning-main/cyber_physic_dataset/split_version/{segment_name}.csv"
    )

    # Save the segment to a CSV file
    segment_data.to_csv(save_path, index=False, header=None)

    print(f"Segment {segment_name} saved to {save_path}")
