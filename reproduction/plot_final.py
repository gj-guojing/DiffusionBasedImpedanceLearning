
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_final_reproduction():
    results_file = "full_paper_results/inference_results.txt"
    if not os.path.exists(results_file):
        print("Error: Full results file not found.")
        return
    
    # Load full data
    df = pd.read_csv(results_file, sep='\t')
    
    # Select a sequence to visualize
    seq_idx = 0
    seq0 = df[df['Seq_Index'] == seq_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # 1. Position X: Full Model Performance
    ax1.plot(seq0['Time'], seq0['Clean_X'], 'g-', label='Clean (ZFT Target)', linewidth=2.5)
    ax1.plot(seq0['Time'], seq0['Noisy_X'], 'r--', label='Noisy (Observed)', alpha=0.3)
    ax1.plot(seq0['Time'], seq0['Denoised_X'], 'b-', label='Denoised (Full Model sZFT)', linewidth=1.5)
    ax1.set_title(f"Full Paper Reproduction (Seq {seq_idx}) - X-axis Trajectory", fontsize=16)
    ax1.set_ylabel("Position [m]", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Position Y: Full Model Performance
    ax2.plot(seq0['Time'], seq0['Clean_Y'], 'g-', label='Clean (ZFT Target)', linewidth=2.5)
    ax2.plot(seq0['Time'], seq0['Noisy_Y'], 'r--', label='Noisy (Observed)', alpha=0.3)
    ax2.plot(seq0['Time'], seq0['Denoised_Y'], 'b-', label='Denoised (Full Model sZFT)', linewidth=1.5)
    ax2.set_title("Trajectory Reconstruction - Y-axis", fontsize=16)
    ax2.set_ylabel("Position [m]", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. External Force Inputs
    ax3.fill_between(seq0['Time'], seq0['Force_X'], color='gray', alpha=0.2, label='Force Area')
    ax3.plot(seq0['Time'], seq0['Force_X'], 'k-', label='External Force X', linewidth=1)
    ax3.plot(seq0['Time'], seq0['Force_Y'], 'blue', label='External Force Y', alpha=0.5, linewidth=1)
    ax3.set_title("External Wrench (Conditioning Signal)", fontsize=16)
    ax3.set_xlabel("Time [s]", fontsize=12)
    ax3.set_ylabel("Force [N]", fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_path = "full_paper_results/final_reproduction_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"✅ Final visualization saved to: {output_path}")

if __name__ == "__main__":
    plot_final_reproduction()
