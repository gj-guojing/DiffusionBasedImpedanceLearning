
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_reproduction():
    results_file = "repro_results/inference_results.txt"
    if not os.path.exists(results_file):
        print("Error: Results file not found.")
        return
    
    # Load data
    df = pd.read_csv(results_file, sep='\t')
    
    # Select the first sequence for visualization
    seq0 = df[df['Seq_Index'] == 0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Position X: Clean vs Noisy vs Denoised
    ax1.plot(seq0['Time'], seq0['Clean_X'], 'g-', label='Clean (ZFT Target)', linewidth=2)
    ax1.plot(seq0['Time'], seq0['Noisy_X'], 'r--', label='Noisy (Observed)', alpha=0.4)
    ax1.plot(seq0['Time'], seq0['Denoised_X'], 'b-', label='Denoised (Reconstructed sZFT)', linewidth=1.5)
    ax1.set_title("Trajectory Reconstruction (X-axis) - Reproduction of Paper Fig. 4", fontsize=14)
    ax1.set_ylabel("Position [m]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Position Y: Clean vs Noisy vs Denoised
    ax2.plot(seq0['Time'], seq0['Clean_Y'], 'g-', label='Clean (ZFT Target)', linewidth=2)
    ax2.plot(seq0['Time'], seq0['Noisy_Y'], 'r--', label='Noisy (Observed)', alpha=0.4)
    ax2.plot(seq0['Time'], seq0['Denoised_Y'], 'b-', label='Denoised (Reconstructed sZFT)', linewidth=1.5)
    ax2.set_title("Trajectory Reconstruction (Y-axis)", fontsize=14)
    ax2.set_ylabel("Position [m]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. External Force (The condition for denoising)
    ax3.plot(seq0['Time'], seq0['Force_X'], 'k-', label='External Force X', alpha=0.8)
    ax3.plot(seq0['Time'], seq0['Force_Y'], 'gray', label='External Force Y', alpha=0.6)
    ax3.set_title("External Wrench Inputs (Cross-Attention Conditioning)", fontsize=14)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Force [N]")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "repro_results/reproduction_plot.png"
    plt.savefig(output_path)
    print(f"Visualization plot saved to: {output_path}")

if __name__ == "__main__":
    plot_reproduction()
