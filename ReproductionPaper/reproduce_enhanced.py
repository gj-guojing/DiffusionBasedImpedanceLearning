
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

print("DEBUG: Loading Enhanced Reproduction Script...")
repo_root = os.path.join(os.getcwd(), "DiffusionBasedImpedanceLearning")
sys.path.append(os.path.join(repo_root, "ImpedanceLearning"))

from models import NoisePredictorTransformerWithCrossAttentionTime
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, inference_simulation
from utils import set_seed

def reproduce_enhanced():
    set_seed(42)
    device = torch.device("cpu")
    print(f"Executing Enhanced Reproduction on {device}...")

    # --- ENHANCED PARAMETERS (Closest possible on CPU) ---
    seq_length = 16
    hidden_dim = 256  # DOUBLED capacity (Paper used 512)
    batch_size = 64
    num_epochs = 5    # INCREASED epochs (Paper used 20)
    learning_rate = 1e-4
    noiseadding_steps = 5
    use_forces = True
    
    data_path = os.path.join(repo_root, "Data/Parkour")
    app_data_path = os.path.join(repo_root, "Data/Parkour/ApplicationData")
    save_path = "enhanced_results"
    os.makedirs(save_path, exist_ok=True)

    # 1. Load Data
    data = load_robot_data(data_path, seq_length, use_overlap=True)
    data_app = load_robot_data(app_data_path, seq_length, use_overlap=False)
    
    # 2. Preprocess
    stats = compute_statistics_per_axis(data)
    normalized_data = normalize_data_per_axis(data, stats)
    normalized_app = normalize_data_per_axis(data_app, stats)
    
    train_size = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:train_size]
    val_data = normalized_data[train_size:]
    
    train_loader = DataLoader(ImpedanceDatasetDiffusion(train_data, stats), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ImpedanceDatasetDiffusion(val_data, stats), batch_size=batch_size, shuffle=False)
    app_loader = DataLoader(ImpedanceDatasetDiffusion(normalized_app, stats), batch_size=1, shuffle=False)

    # 3. Model
    model = NoisePredictorTransformerWithCrossAttentionTime(
        seq_length=seq_length, 
        hidden_dim=hidden_dim, 
        num_timesteps=noiseadding_steps,
        use_forces=use_forces
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 4. Training
    print(f"Training Enhanced Model (Dim {hidden_dim}, Epochs {num_epochs})...")
    train_model_diffusion(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=num_epochs,
        noiseadding_steps=noiseadding_steps,
        beta_start=0.0001,
        beta_end=0.04,
        use_forces=use_forces,
        save_path=save_path,
        early_stop_patience=3
    )

    # 5. Inference
    print("Running Final Inference Simulation...")
    inference_simulation(
        model, app_loader, ImpedanceDatasetDiffusion(normalized_app, stats), 
        device, use_forces, save_path, num_sequences=10, num_denoising_steps=5
    )
    print("Enhanced Reproduction Complete.")

if __name__ == "__main__":
    reproduce_enhanced()
