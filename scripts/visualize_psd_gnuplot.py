import os
import torch
import subprocess
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

def main():
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    seq_len = 5000  # Longer sequence for smoother PSD
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    os.makedirs("noise_plot", exist_ok=True)
    data_file = "noise_plot/psd_data.dat"
    with open(data_file, 'w') as f:
        noises = []
        psds = []
        freqs = None
        
        for beta in betas:
            # Average over a few sequences to get a smoother periodogram
            psd_avg = None
            for _ in range(5):
                noise_tensor = generate_colored_noise([1, 1, seq_len], device, dtype, exponent=beta)
                noise_array = noise_tensor.squeeze().cpu().numpy()
                f_welch, p_welch = signal.welch(noise_array, fs=1.0, nperseg=512)
                
                if psd_avg is None:
                    psd_avg = p_welch
                    freqs = f_welch
                else:
                    psd_avg += p_welch
            
            psd_avg /= 5.0
            psds.append(psd_avg)
            
        # Write to columns: freq, p1, p2, p3, p4, p5 (skip DC component f=0 to avoid log(0) issues)
        for i in range(1, len(freqs)):
            row = f"{freqs[i]:.6f}\t" + "\t".join(f"{p[i]:.6e}" for p in psds)
            f.write(row + "\n")
            
    # Gnuplot script
    plt_cmds = """
set title "Power Spectral Density (PSD) of Colored Observation Noise" font "Times New Roman, 16"

set logscale xy
set grid lc rgb "#e0e0e0" lw 1 lt 0

set xlabel "Frequency (Hz)" font "Times New Roman, 14"
set ylabel "Power / Frequency" font "Times New Roman, 14"
set key bottom left spacing 1.5 font "Times New Roman, 12"

# Set decent ranges
set yrange [1e-5:1e4]

plot "noise_plot/psd_data.dat" using 1:2 title "White Noise ({/Symbol b}=0.0)" with lines lc rgb "#333333" lw 2, \\
     "noise_plot/psd_data.dat" using 1:3 title "Colored Noise ({/Symbol b}=0.5)" with lines lc rgb "#228B22" lw 2, \\
     "noise_plot/psd_data.dat" using 1:4 title "Pink Noise ({/Symbol b}=1.0)" with lines lc rgb "#8A2BE2" lw 2, \\
     "noise_plot/psd_data.dat" using 1:5 title "Colored Noise ({/Symbol b}=1.5)" with lines lc rgb "#FF8C00" lw 2, \\
     "noise_plot/psd_data.dat" using 1:6 title "Brown Noise ({/Symbol b}=2.0)" with lines lc rgb "#DC143C" lw 2
"""

    plt_file = "noise_plot/plot_psd.plt"
    with open(plt_file, 'w') as f:
        f.write('set terminal pngcairo size 800,600 font "Times New Roman,12" enhanced\n')
        f.write('set output "noise_plot/psd_comparison.png"\n')
        f.write(plt_cmds)
        
        f.write('\nset terminal pdfcairo size 6,4.5 font "Times New Roman,12" enhanced\n')
        f.write('set output "noise_plot/psd_comparison.pdf"\n')
        f.write(plt_cmds)
        
    print("Running gnuplot...")
    subprocess.run(["gnuplot", plt_file])
    print("Generated noise_plot/psd_comparison.png and noise_plot/psd_comparison.pdf")

if __name__ == "__main__":
    main()
