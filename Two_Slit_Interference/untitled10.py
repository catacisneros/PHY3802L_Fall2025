import LT.box as B 
import numpy as np 


L_s = B.get_file('left_slit.data.dat')
V = L_s['V']
S = L_s['S']
B.plot_exp(S,V)
B.plot_line(S,V)
                             

B.pl.figure()
R_s = B.get_file('right_slit.data.dat')
V = R_s['V']
S = R_s['S']
B.plot_exp(S,V)
B.plot_line(S,V)

B.pl.figure()
D_s = B.get_file('two_slit.data.dat')
V = D_s['V']
S = D_s['S']
B.plot_exp(S,V)
B.plot_line(S,V) 