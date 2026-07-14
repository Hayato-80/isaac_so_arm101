set terminal pngcairo size 800,600 font "Times New Roman,12" enhanced
set output "noise_plot/psd_comparison.png"

set title "Power Spectral Density (PSD) of Colored Observation Noise" font "Times New Roman, 16"

set logscale xy
set grid lc rgb "#e0e0e0" lw 1 lt 0

set xlabel "Frequency (Hz)" font "Times New Roman, 14"
set ylabel "Power / Frequency" font "Times New Roman, 14"
set key bottom left spacing 1.5 font "Times New Roman, 12"

# Set decent ranges
set yrange [1e-5:1e4]

plot "noise_plot/psd_data.dat" using 1:2 title "White Noise ({/:Italic β}=0.0)" with lines lc rgb "#333333" lw 2, \
     "noise_plot/psd_data.dat" using 1:3 title "Colored Noise ({/:Italic β}=0.5)" with lines lc rgb "#228B22" lw 2, \
     "noise_plot/psd_data.dat" using 1:4 title "Pink Noise ({/:Italic β}=1.0)" with lines lc rgb "#FF00FF" lw 2, \
     "noise_plot/psd_data.dat" using 1:5 title "Colored Noise ({/:Italic β}=1.5)" with lines lc rgb "#FF8C00" lw 2, \
     "noise_plot/psd_data.dat" using 1:6 title "Red Noise ({/:Italic β}=2.0)" with lines lc rgb "#DC143C" lw 2

set terminal pdfcairo size 6,4.5 font "Times New Roman,12" enhanced
set output "noise_plot/psd_comparison.pdf"

set title "Power Spectral Density (PSD) of Colored Observation Noise" font "Times New Roman, 16"

set logscale xy
set grid lc rgb "#e0e0e0" lw 1 lt 0

set xlabel "Frequency (Hz)" font "Times New Roman, 14"
set ylabel "Power / Frequency" font "Times New Roman, 14"
set key bottom left spacing 1.5 font "Times New Roman, 12"

# Set decent ranges
set yrange [1e-5:1e4]

plot "noise_plot/psd_data.dat" using 1:2 title "White Noise ({/:Italic β}=0.0)" with lines lc rgb "#333333" lw 2, \
     "noise_plot/psd_data.dat" using 1:3 title "Colored Noise ({/:Italic β}=0.5)" with lines lc rgb "#228B22" lw 2, \
     "noise_plot/psd_data.dat" using 1:4 title "Pink Noise ({/:Italic β}=1.0)" with lines lc rgb "#FF00FF" lw 2, \
     "noise_plot/psd_data.dat" using 1:5 title "Colored Noise ({/:Italic β}=1.5)" with lines lc rgb "#FF8C00" lw 2, \
     "noise_plot/psd_data.dat" using 1:6 title "Red Noise ({/:Italic β}=2.0)" with lines lc rgb "#DC143C" lw 2
