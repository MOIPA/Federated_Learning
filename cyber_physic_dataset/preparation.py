import pandas as pd

# Load your dataset
dataset_path = './cyber_physic_dataset/Dataset.csv'  # Update this path to your actual dataset location
data = pd.read_csv(dataset_path)

# Define the segments for the 10 categories
segments = {
    "Benign_Cyber": (1, 9426),
    "Benign_Physical": (9427, 13717),
    "DoS_Attack_Cyber": (13718, 25389),
    "DoS_Attack_Physical": (25390, 26363),
    "Replay_Attack_Cyber": (26364, 38370),
    "Replay_Attack_Physical": (38371, 39344),
    "Evil_Twin_Cyber": (39345, 45028),
    "Evil_Twin_Physical": (45029, 50502),
    "FDI_Cyber": (50503, 53976),
    "FDI_Physical": (53977, 54784)
}

# Segment the dataset and save each segment to a new CSV file
for segment_name, (start, end) in segments.items():
    # Adjust for zero-based indexing
    segment_data = data.iloc[start-1:end]
    
    # Define the path for the new CSV file
    # Replace 'your_directory_path' with the path where you want to save the files
    save_path = f'./cyber_physic_dataset/{segment_name}.csv'
    
    # Save the segment to a CSV file
    segment_data.to_csv(save_path, index=False)

    print(f"Segment {segment_name} saved to {save_path}")
