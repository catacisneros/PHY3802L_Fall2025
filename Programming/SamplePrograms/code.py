#Practicing fitting plos with various functions import numpy as np

import numpy as np
import LT.box as B

file_name = "examples_data"
f = B.get_file(file_name) #retrieves data from the database and saves it in 'f'
#get the data
#current

A = B.get_data(f, 'A') #uses LT.box library (called B) to retrieve the data from the data base f (previouslt imported the whole database) specified column A
b = B.get_data(f, 'b')
db = B.get_data(f, 'db')
C = B.get_data(f, 'C')
D = B.get_data(f, 'D')

#the following examples fit1 to fit4 fits C vs A, using linear fit, 
#polynomial fits, in either the whole ranges (fit1, fit2) of a 
#subrange (fit 3, 4)

B.plot_exp(A, C, db) #creates plot as an exponential function
B.pl.show() #shows it

#you should uncomment the next line to see what it does to the x-axis
#Can you set the y-axis title now?

B.pl.xlabel("x (unit)") #labels the x axis with "x (unit)", B, plotting object (pl)
fit1 = B.linefit(A, C, db)  #linear regression (line fitting) of A(x) and C(y), where db might be an initial guess or fit setting
# fit1 = B.linefit(A, C, db)
# chisq/dof = 1.7887   -> goodness of fit (close to 1 = good)
# offset = -8.8868 ± 0.2444 -> y-intercept of the line
# slope  = 9.4636 ± 0.1290  -> slope of the line

B.plot_line(fit1.xpl, fit1.ypl) #draws the plot of the fitted object (above) xpl(x-axis), ypl(y-axis)

#the following 2 lines selecting the ranges

r1 = B.in_between(4.0, 10.0, C) #in_window to be changed to in_between
r2 = B.in_between(1.0, 12.0, C)

#in the terminal (console)
fit2 = B.polyfit(A, C, db) #polynomial default is 2
fit2 = B.polyfit(A, C, db, 5) #setting a polynomial to 5
fit2.parameters[4].value #displays the 4th value of the previous polynomial fitting
#fit2.parameters[4].
# can be: value, err, get, name, set... can be found out by starting to type on the console and press 'tab' (guess first few letters or guess lol)
#chi-squared is how good of a fit it is. better --> near to one. x^2 = sum of (f(x1)-y1)^2
# fitting parameter value, uncertainy of the parameter, and the reduced chi squared


## fitting can be done with any function: linear, polynomial, trigonometric (sin, cos), exponential... 

#only fit the selected ranges as specified by r1 and r2
fit3 = B.linefit(A[r1], C[r1], db[r1])
B.plot_line(fit3.xpl, fit3.ypl)

#after fitting is done, we must interpret the results 


#the fit below use second order-- defined by the "2" in the arg polynomial
#you should change 2 to other integers and see how the fits are different 
fit4 = 






