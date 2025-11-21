set terminal png
set output 'mtf_curves.png'
set title 'Slant-Edge MTF'
set xlabel 'Frequency'
set ylabel 'MTF'
set grid
plot 'mtf_0.txt' with lines title 'Edge 1', 'mtf_1.txt' with lines title 'Edge 2'
