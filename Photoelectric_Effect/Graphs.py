import LT.box as B
import numpy as np



colors = ['yellow', 'green', 'blue', 'violet', 'uv']

#--- YELLOW ---

B.pl.figure()

#import data


yellow_data = 'Data_yellow.data'
yellow_data = B.get_file(yellow_data)


#define variables
v_yellow = B.get_data(yellow_data, 'V')
i_yellow = B.get_data(yellow_data, 'I')

#plot
B.plot_exp(v_yellow, i_yellow)
B.pl.xlabel('Voltage')
B.pl.ylabel('Current')
B.pl.title('Photoelectric effect for yellow')



# --- GREEN ---

B.pl.figure()

#import data
green_data = 'Data_green.data'
green_data = B.get_file(green_data)


#define variables
v_green = B.get_data(green_data, 'V')
i_green = B.get_data(green_data, 'I')

#plot
B.plot_exp(v_green, i_green)
B.pl.xlabel('Voltage')
B.pl.ylabel('Current')
B.pl.title('Photoelectric effect for green')

# --- BLUE ---

B.pl.figure()

#import data
blue_data = 'Data_blue.data'
blue_data = B.get_file(blue_data)


#define variables
v_blue = B.get_data(blue_data, 'V')
i_blue = B.get_data(blue_data, 'I')

#plot
B.plot_exp(v_blue, i_blue)
B.pl.xlabel('Voltage')
B.pl.ylabel('Current')
B.pl.title('Photoelectric effect for blue')


# --- VIOLET ---

B.pl.figure()

#import data
violet_data = 'Data_violet.data'
violet_data = B.get_file(violet_data)


#define variables
v_violet = B.get_data(violet_data, 'V')
i_violet = B.get_data(violet_data, 'I')

#plot
B.plot_exp(v_violet, i_violet)
B.pl.xlabel('Voltage')
B.pl.ylabel('Current')
B.pl.title('Photoelectric effect for violet')

# --- UV ---

B.pl.figure()

#import data
uv_data = 'Data_uv.data'
uv_data = B.get_file(uv_data)


#define variables
v_uv = B.get_data(uv_data, 'V')
i_uv = B.get_data(uv_data, 'I')

#plot
B.plot_exp(v_uv, i_uv)
B.pl.xlabel('Voltage')
B.pl.ylabel('Current')
B.pl.title('Photoelectric effect for ultra-violet')

# # --- eyeballed --- 

B.pl.figure()

vs = np.array([0.715, 0.773, 0.952, 1.030, 1.551])
w_length = np.array([578, 546, 436, 405, 365])
one_lambda = np.array (1 / np.array(w_length))

B.plot_exp(one_lambda, vs)