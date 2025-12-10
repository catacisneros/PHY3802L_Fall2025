
import LT.box as B 
import numpy as np
import LT_Fit.parameters as P  # get the parameter module
import LT_Fit.gen_fit as G     # load the genfit module
import scipy.special as ms     #need this for the factorial function



#Create an array with random ages
age=([23,19,25,20,18,17,26,30,17,21,22,25,27,7,37,47,19,24,25,20,22,
      30,20,26,27,20,])

#variable to create a histogram including the array age within the range 
#(5, 50) separated in 7 different 'bins'
#a1=B.histo(age, (5,50), 5)
#Plot the histogoroam 
#a1.plot()

#Separate in 10 bins instead of 5
#a2=B.histo(age, (5,50), 10)
#plot
#a2.plot()

#Separate in 40 bins instead 
a3=B.histo(age, (5,40), 40)
#plot
a3.plot()


#### fitting
#put the bin centers and bin contents to two arrays
#hx=a3.bin_content
hx=a3.bin_center
hy=a3.bin_content
unc = np.sqrt(np.maximum(hy, 1.0)) #uncertainty

# #fit histogram polynomial
# fith=B.polyfit( hx, hy, order=10)
# B.plot_line(fith.xpl, fith.ypl)

# #regiular fit
# fith = a3.fit()


age2=([23,19,25,20,18,17,26,30,17,21,22,25,27,7,37,47,19,24,25,20,22,
      30,20,26,27,20,47,45, 39, 45, 23, 36, 44, 36,24,33,56,32,49,35,35,45,
      54,49,56,75,42,63,45,34,36,48,48,38,24,52,34,24,62,54,32,36,45,34])


#Separate in 40 bins instead 
a4=B.histo(age2, (5,60), 60)
#plot
a4.plot()

#### fitting
#put the bin centers and bin contents to two arrays
#hx=a3.bin_content
hx2=a4.bin_center
hy2=a4.bin_content
unc2 = np.sqrt(np.maximum(hy2, 1.0)) #uncertainty

#fit histogram Gaussian

#parameters
p1=P.Parameter(25, 'p1') #median
p2=P.Parameter(22, 'p2')
p3=P.Parameter(6, 'p3')

#define the gaussian function
def gauss(x):
    return p1() * np.exp(-0.5 * ((x - p2()) / p3())**2)

#new fitted with gauss variable 
fit = G.genfit(gauss, [p1, p2, p3], x=hx, y=hy, y_err=unc)

fit.fit()
fit.show_results()

B.pl.plot_line(hx, gauss(hx))





