import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def generate_colored_noise(
    size: list[int] | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    exponent: float = 1.0,
) -> torch.Tensor:
    samples = size[-1]
    f = torch.fft.rfftfreq(samples, device=device)
    fmin = 1.0 / samples
    s_scale = f.clone()
    cutoff_idx = torch.sum(s_scale < fmin).item()
    if cutoff_idx > 0 and cutoff_idx < len(s_scale):
        s_scale[:cutoff_idx] = s_scale[cutoff_idx]
    s_scale = s_scale ** (-exponent / 2.0)
    w = s_scale[1:].clone()
    w[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2.0 * torch.sqrt(torch.sum(w ** 2)) / samples
    noise_size = list(size)
    noise_size[-1] = len(f)
    s_scale_broadcast = s_scale.view(*([1] * (len(size) - 1)), -1)
    sr = torch.randn(noise_size, device=device, dtype=dtype) * s_scale_broadcast
    si = torch.randn(noise_size, device=device, dtype=dtype) * s_scale_broadcast
    if not (samples % 2):
        si[..., -1] = 0.0
        sr[..., -1] *= 1.4142135623730951
    si[..., 0] = 0.0
    sr[..., 0] *= 1.4142135623730951
    s = torch.complex(sr, si)
    y = torch.fft.irfft(s, n=samples, dim=-1) / sigma
    return y

def plot_noise_trajectories_and_psd():
    # Simulation parameters matching training
    num_envs = 1
    obs_dim = 1
    seq_len = 1000 # Typical episode length
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Betas to visualize
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    labels = ["White Noise (beta=0.0)", "beta=0.5", "Pink Noise (beta=1.0)", "beta=1.5", "Brown/Red Noise (beta=2.0)"]
    colors = ['gray', 'green', 'magenta', 'orange', 'red']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for beta, label, color in zip(betas, labels, colors):
        # Generate noise using the exact function from training
        # generate_colored_noise returns shape [..., time_steps]
        size = [num_envs, obs_dim, seq_len]
        noise_tensor = generate_colored_noise(size, device, dtype, exponent=beta)
        
        # Squeeze to 1D array and move to CPU for plotting
        noise_array = noise_tensor.squeeze().cpu().numpy()

        # Plot time domain trajectory
        time_steps = np.arange(seq_len)
        # Offset the trajectories slightly for better visibility in time domain
        offset = beta * 3
        ax1.plot(time_steps, noise_array + offset, label=f"{label} (offset {offset})", color=color, alpha=0.8)

        # Plot Power Spectral Density (PSD) using Welch's method
        freqs, psd = signal.welch(noise_array, fs=1.0, nperseg=256)
        
        # Avoid log(0) issues by slicing from 1
        ax2.loglog(freqs[1:], psd[1:], label=label, color=color, alpha=0.8)

        # Plot theoretical 1/f^beta line for reference (scaled to match PSD roughly)
        if beta > 0:
            theoretical_psd = 1.0 / (freqs[1:] ** beta)
            # Normalize to match the first point for visual comparison
            theoretical_psd *= psd[1] / theoretical_psd[0]
            ax2.loglog(freqs[1:], theoretical_psd, color=color, linestyle='--', alpha=0.5)

    ax1.set_title("Time Domain: Generated Noise Trajectories")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Noise Value (with arbitrary offset)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Frequency Domain: Power Spectral Density (PSD)")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Power")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'noise_plot')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'noise_visualization.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_noise_trajectories_and_psd()
