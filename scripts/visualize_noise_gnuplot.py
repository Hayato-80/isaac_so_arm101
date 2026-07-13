import os
import torch
import subprocess

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
    seq_len = 1000
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    os.makedirs("noise_plot", exist_ok=True)
    data_file = "noise_plot/noise_data.dat"
    with open(data_file, 'w') as f:
        # Generate all noises
        noises = []
        for beta in betas:
            noise_tensor = generate_colored_noise([1, 1, seq_len], device, dtype, exponent=beta)
            noises.append(noise_tensor.squeeze().cpu().numpy())
            
        # Write to columns: step, n1, n2, n3, n4, n5
        for i in range(seq_len):
            row = f"{i}\t" + "\t".join(f"{n[i]:.6f}" for n in noises)
            f.write(row + "\n")
            
    multiplot_cmds = """
set multiplot layout 5,1 title "Colored Observation Noise Trajectories" font "Times New Roman, 16" margins 0.12, 0.95, 0.1, 0.92 spacing 0.0, 0.08

unset key
set grid ytics lc rgb "#e0e0e0" lw 1 lt 0
set grid xtics lc rgb "#e0e0e0" lw 1 lt 0

set xrange [0:1000]
set ylabel "Value" offset 1,0 font "Times New Roman,12"

set format x ""
set title "White Noise ({/:Italic β}=0.0)" font "Times New Roman,12"
plot "noise_plot/noise_data.dat" using 1:2 with lines lc rgb "#333333" lw 1.2

set title "Colored Noise ({/:Italic β}=0.5)"
plot "noise_plot/noise_data.dat" using 1:3 with lines lc rgb "#228B22" lw 1.2

set title "Pink Noise ({/:Italic β}=1.0)"
plot "noise_plot/noise_data.dat" using 1:4 with lines lc rgb "#8A2BE2" lw 1.2

set title "Colored Noise ({/:Italic β}=1.5)"
plot "noise_plot/noise_data.dat" using 1:5 with lines lc rgb "#FF8C00" lw 1.2

set format x "%g"
set xlabel "Time Step" font "Times New Roman,12"
set title "Brown/Red Noise ({/:Italic β}=2.0)"
plot "noise_plot/noise_data.dat" using 1:6 with lines lc rgb "#DC143C" lw 1.2

unset multiplot
"""

    plt_file = "noise_plot/plot_noise.plt"
    with open(plt_file, 'w') as f:
        f.write('set terminal pngcairo size 1000,1000 font "Times New Roman,14" enhanced\n')
        f.write('set output "noise_plot/noise_comparison.png"\n')
        f.write(multiplot_cmds)
        
        f.write('\nset terminal pdfcairo size 6,6 font "Times New Roman,10" enhanced\n')
        f.write('set output "noise_plot/noise_comparison.pdf"\n')
        f.write(multiplot_cmds)
        
    print("Running gnuplot...")
    subprocess.run(["gnuplot", plt_file])
    print("Generated noise_plot/noise_comparison.png and noise_plot/noise_comparison.pdf")

if __name__ == "__main__":
    main()
