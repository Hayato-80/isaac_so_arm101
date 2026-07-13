set terminal pngcairo size 1000,1000 font "Times New Roman,14" enhanced
set output "noise_plot/noise_comparison.png"

set multiplot layout 5,1 title "Colored Observation Noise Trajectories" font "Times New Roman, 16" margins 0.12, 0.95, 0.1, 0.92 spacing 0.0, 0.08

unset key
set grid ytics lc rgb "#e0e0e0" lw 1 lt 0
set grid xtics lc rgb "#e0e0e0" lw 1 lt 0

set xrange [0:1000]
set ylabel "Value" offset 1,0 font "Times New Roman,12"

set format x ""
set title "White Noise ({/Symbol b}=0.0)" font "Times New Roman,12"
plot "noise_plot/noise_data.dat" using 1:2 with lines lc rgb "#333333" lw 1.2

set title "Colored Noise ({/Symbol b}=0.5)"
plot "noise_plot/noise_data.dat" using 1:3 with lines lc rgb "#228B22" lw 1.2

set title "Pink Noise ({/Symbol b}=1.0)"
plot "noise_plot/noise_data.dat" using 1:4 with lines lc rgb "#8A2BE2" lw 1.2

set title "Colored Noise ({/Symbol b}=1.5)"
plot "noise_plot/noise_data.dat" using 1:5 with lines lc rgb "#FF8C00" lw 1.2

set format x "%g"
set xlabel "Time Step" font "Times New Roman,12"
set title "Brown/Red Noise ({/Symbol b}=2.0)"
plot "noise_plot/noise_data.dat" using 1:6 with lines lc rgb "#DC143C" lw 1.2

unset multiplot

set terminal pdfcairo size 6,6 font "Times New Roman,10" enhanced
set output "noise_plot/noise_comparison.pdf"

set multiplot layout 5,1 title "Colored Observation Noise Trajectories" font "Times New Roman, 16" margins 0.12, 0.95, 0.1, 0.92 spacing 0.0, 0.08

unset key
set grid ytics lc rgb "#e0e0e0" lw 1 lt 0
set grid xtics lc rgb "#e0e0e0" lw 1 lt 0

set xrange [0:1000]
set ylabel "Value" offset 1,0 font "Times New Roman,12"

set format x ""
set title "White Noise ({/Symbol b}=0.0)" font "Times New Roman,12"
plot "noise_plot/noise_data.dat" using 1:2 with lines lc rgb "#333333" lw 1.2

set title "Colored Noise ({/Symbol b}=0.5)"
plot "noise_plot/noise_data.dat" using 1:3 with lines lc rgb "#228B22" lw 1.2

set title "Pink Noise ({/Symbol b}=1.0)"
plot "noise_plot/noise_data.dat" using 1:4 with lines lc rgb "#8A2BE2" lw 1.2

set title "Colored Noise ({/Symbol b}=1.5)"
plot "noise_plot/noise_data.dat" using 1:5 with lines lc rgb "#FF8C00" lw 1.2

set format x "%g"
set xlabel "Time Step" font "Times New Roman,12"
set title "Brown/Red Noise ({/Symbol b}=2.0)"
plot "noise_plot/noise_data.dat" using 1:6 with lines lc rgb "#DC143C" lw 1.2

unset multiplot
