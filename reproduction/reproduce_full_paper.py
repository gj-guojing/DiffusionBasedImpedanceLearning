
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# --- 0. Environment Setup ---
print("=== Starting Full Paper Reproduction (No Compromise) ===")
repo_root = os.path.join(os.getcwd(), "DiffusionBasedImpedanceLearning")
sys.path.append(os.path.join(repo_root, "ImpedanceLearning"))

from models import NoisePredictorTransformerWithCrossAttentionTime
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, inference_simulation
from utils import set_seed

def reproduce_full_paper():
    # Set seed for reproducibility
    set_seed(42)

    # --- 1. Hardware Check ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print("   Ready for full-scale training.")
    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: No GPU detected. Running on CPU.")
        print("   This 'No Compromise' configuration will take extremely long (10+ hours) on CPU.")
    
    # --- 2. Paper Parameters (Golden Standard) ---
    config = {
        "seq_length": 16,
        "hidden_dim": 512,        # Original Paper Dimension
        "batch_size": 64,
        "num_epochs": 20,         # Original Paper Epochs
        "learning_rate": 1e-4,
        "noiseadding_steps": 5,   
        "use_forces": True,
        "beta_start": 0.0001,
        "beta_end": 0.04
    }
    
    print(f"Configuration: {config}")

    # --- 3. Data Loading (Full Dataset) ---
    data_path = os.path.join(repo_root, "Data/Parkour")
    app_data_path = os.path.join(repo_root, "Data/Parkour/ApplicationData")
    save_path = "full_paper_results"
    os.makedirs(save_path, exist_ok=True)

    print("Loading FULL dataset (approx 50k+ samples)...")
    data = load_robot_data(data_path, config["seq_length"], use_overlap=True)
    data_app = load_robot_data(app_data_path, config["seq_length"], use_overlap=False)
    
    print("Computing global statistics...")
    stats = compute_statistics_per_axis(data)
    normalized_data = normalize_data_per_axis(data, stats)
    normalized_app = normalize_data_per_axis(data_app, stats)
    
    # Split Data
    train_size = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:train_size]
    val_data = normalized_data[train_size:]
    
    # Use num_workers=0 to avoid potential multiprocessing issues in some envs
    train_loader = DataLoader(ImpedanceDatasetDiffusion(train_data, stats), 
                              batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(ImpedanceDatasetDiffusion(val_data, stats), 
                             batch_size=config["batch_size"], shuffle=False, num_workers=0)
    app_loader = DataLoader(ImpedanceDatasetDiffusion(normalized_app, stats), 
                            batch_size=1, shuffle=False)

    # --- 4. Model Initialization ---
    print(f"Initializing Transformer Model (Hidden Dim: {config['hidden_dim']})...")
    model = NoisePredictorTransformerWithCrossAttentionTime(
        seq_length=config["seq_length"], 
        hidden_dim=config["hidden_dim"], 
        num_timesteps=config["noiseadding_steps"],
        use_forces=config["use_forces"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # --- 5. Full Training ---
    print("Starting Training Loop...")
    train_model_diffusion(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=config["num_epochs"],
        noiseadding_steps=config["noiseadding_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        use_forces=config["use_forces"],
        save_path=save_path,
        early_stop_patience=5
    )

    # --- 6. Full Inference ---
    print("Running Full Inference Simulation...")
    inference_simulation(
        model, app_loader, ImpedanceDatasetDiffusion(normalized_app, stats), 
        device, config["use_forces"], save_path, num_sequences=100, num_denoising_steps=5
    )
    print(f"Full Reproduction Complete. Results saved to {save_path}")

if __name__ == "__main__":
    reproduce_full_paper()
