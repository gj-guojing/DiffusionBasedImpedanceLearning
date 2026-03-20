
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gc
from torch.utils.data import DataLoader
from datetime import datetime

print("DEBUG: Script started. Setting up paths...")
# Import components from the repository
repo_root = os.path.join(os.getcwd(), "DiffusionBasedImpedanceLearning")
sys.path.append(os.path.join(repo_root, "ImpedanceLearning"))

try:
    print("DEBUG: Importing project modules...")
    from models import NoisePredictorTransformerWithCrossAttentionTime
    from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
    from train_val_test import train_model_diffusion, test_model, inference_simulation
    from utils import set_seed
    print("DEBUG: Imports successful.")
except Exception as e:
    print(f"DEBUG ERROR during imports: {e}")
    sys.exit(1)

def reproduce():
    print("Starting Minimal Reproduction of 'Diffusion-Based Impedance Learning'...")
    set_seed(42)
    device = torch.device("cpu") # Force CPU for predictable reproduction
    print(f"Using device: {device}")

    # --- 1. Setup Parameters (Super minimal) ---
    seq_length = 16
    hidden_dim = 128  # Even smaller
    batch_size = 32
    num_epochs = 2    # Just 2 epochs
    learning_rate = 1e-4
    noiseadding_steps = 5
    use_forces = True
    
    beta_start = 0.0001
    beta_end = 0.04
    
    data_path = os.path.join(repo_root, "Data/Parkour")
    app_data_path = os.path.join(repo_root, "Data/Parkour/ApplicationData")
    save_path = "repro_results"
    os.makedirs(save_path, exist_ok=True)

    # --- 2. Load and Preprocess Data ---
    print(f"Loading training data from {data_path}...")
    try:
        data = load_robot_data(data_path, seq_length, use_overlap=True)
        print(f"Loaded {len(data)} samples.")
        data_app = load_robot_data(app_data_path, seq_length, use_overlap=False)
        print(f"Loaded {len(data_app)} application samples.")
    except Exception as e:
        print(f"DEBUG ERROR during data loading: {e}")
        return

    print("Computing statistics...")
    stats = compute_statistics_per_axis(data)
    normalized_data = normalize_data_per_axis(data, stats)
    normalized_app = normalize_data_per_axis(data_app, stats)
    
    train_size = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:train_size]
    val_data = normalized_data[train_size:]
    
    train_loader = DataLoader(ImpedanceDatasetDiffusion(train_data, stats), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ImpedanceDatasetDiffusion(val_data, stats), batch_size=batch_size, shuffle=False)
    app_loader = DataLoader(ImpedanceDatasetDiffusion(normalized_app, stats), batch_size=1, shuffle=False)

    # --- 3. Initialize Model ---
    print("Initializing model...")
    model = NoisePredictorTransformerWithCrossAttentionTime(
        seq_length=seq_length, 
        hidden_dim=hidden_dim, 
        num_timesteps=noiseadding_steps,
        use_forces=use_forces
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- 4. Training ---
    print(f"Training for {num_epochs} epochs...")
    train_model_diffusion(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=num_epochs,
        noiseadding_steps=noiseadding_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        use_forces=use_forces,
        save_path=save_path,
        early_stop_patience=2
    )

    # --- 5. Inference Simulation ---
    print("Running Inference Simulation...")
    inference_simulation(
        model, app_loader, ImpedanceDatasetDiffusion(normalized_app, stats), 
        device, use_forces, save_path, num_sequences=2, num_denoising_steps=5
    )

    print("Reproduction complete. Results in 'repro_results'.")

if __name__ == "__main__":
    reproduce()
