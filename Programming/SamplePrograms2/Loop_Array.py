import LT.box as B 
import numpy as np

# a lot of times when you are dealing with a few data points, you don't
# haveto read them from a datafile . You can quickly define using the
# np.array() -

x=np.array([1, 2, 3, 4]) # attention to the parenthesis out and bracket inside
#creating variable x with an array range from 1 to 4 (int)

y=np.array([21,4.0,5.9,7.5])
#creating a variable y with an array of float

deltay=np.array([0.1,0.1,0.2,0.2])
#creating another variable deltay containing an array of float numbers

B.plot_exp(x,y,dy=deltay)
#Now you made a plot very quickly without creating a data file first and
# then reading from it
#the data were the variables x and y, using deltay as the error margin


################
#What if you have multiple data files which you want to analyze similarly
#then youll benefit from using loops
#for example we have three data files (yellow green and blue) with similar def
# we can do the following

filenames=np.array(['yellow.dat', 'green.dat', 'blue.dat'])
#creating an array of the data files names (not the data contained in them)

range_low=np.array([1,2,3])
range_high=np.array([8,9,10])
#creating two more arrays saved in the variables above said


#since there are 3 daa files, the argument to the function range() below
#should be range [0,3]

for fileindex in range(0,3):
#instead of (1,3) you could also use range(3). fileindex is an int starting 
#from 0 ending in 2.

    print ('fileindex is', fileindex)
    #prints each file index once. total of 3 times for 0, 1 and 2
    f = B.get_file(filenames[fileindex])
    #load the data file in the same order as it prints the file index (prev)
    print(filenames[fileindex])
    #prints the name of each data file in the same order as prev
    
    A=B.get_data(f,'A')
    #load the data from the index 'f' in order which is either 0, 1, or 2 
    #corresponding to each of the loaded files and loads the column specified 
    #as A in the dat file
    print(A)
    #prints the data of the column A of the file being read in the loop 
    
    b=B.get_data(f,'b')
    db=B.get_data(f,'db')
    # loads data from the columns 'b' and 'db' of the dat files in 
    #order of the index
    
    B.pl.figure() 
    #this creates a new figure window for the plot. all the graphs will 
    #be in the same window
    
    B.plot_exp(A, b, dy=db)
    #creates an exponential graph with x=A, y=B and the uncertainty=db
    
    myrange=B.in_between(range_low[fileindex], range_high[fileindex],A)
    #chooses which data from A is inside the range specified (above)
    
    fit=B.linefit(A[myrange], b[myrange], db[myrange])
    #This fits a straight line to the data points, but only uses the data that 
    #falls within the range we specified earlier (myrange), not all the data
    #not a plot, only the fit
    
    B.plot_line(fit.xpl, fit.ypl)
    #adding the prev fitted line to the plot
    
    
   #the above define the fitting window in terms of variable A, using
# the previously defined two arrays: range_low and range_high
#don't forget that the [fileindex] argument; see comments below
#What if you want to fit the graphs, but with a different range
# then you need to define the ranges (both lower and upper limits)
# in array as well, before the loop
# similarly, if you were to fit those data with different initial
# parameters, they should be pre-defined in arrays before the loop
# then access the corresponding array members inside the loop
# Notice that these lines should be indented; if there is no
#If you did everything correctly, you should see three graphs,
#almost straight lines, plotted on top of each other
# what if you want each graph to be plotted on a different canvas?

#############