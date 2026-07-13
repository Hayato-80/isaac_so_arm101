set terminal pngcairo size 400,300 font "Times New Roman,14" enhanced
set output "test_font.png"
set label 1 "1: {/Symbol b}" at 50, 80
set label 2 "2: {/Symbol-Oblique b}" at 50, 60
set label 3 "3: {/:Italic {/Symbol b}}" at 50, 40
set label 4 "4: {/:Italic β}" at 50, 20
set xrange [0:100]
set yrange [0:100]
plot NaN notitle
