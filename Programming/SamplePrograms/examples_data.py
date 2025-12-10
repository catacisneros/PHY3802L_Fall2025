# %%
#\ C0=0.2
# in th line above, the "#\" means what follows is a constant parameter
# you can access this parameter by using the function 
#par.get_value('C0')  refer to the FittingExamples.py to see how; 


#five columns of made up data, notice how each varaible start with the 
#variable name, followed by the varaible type (in this example they are
# all "f", meaning floating number, and index in bracket; The index 
#starts from 0, i.e., the 0th indexed varaible is the first variable; 
#The bracket must be followed by "/", indicating the ending of the 
#definition of that variable.  
# Then each variable is separated by a white space, which could be 
#space(s), or tab(s) 
 
 
#! A[f,0]/ b[f,1]/ db[f,2]/ C[f,3]/ D[f,4]/
#notice how these varaible definitions start with "#!", not "#", or "#\"
    
1.00   11.0   .25   1.0     1.0
1.12    9.5   .25   2.0     1.5
1.24    8.25  .25   3.0     2.5
1.36    7.4   .25   4.0     3.5
1.48    6.95  .25   5.0     5.0
1.60    6.5   .25   6.0     7.0
1.72    6.1   .25   7.0     5.5
1.84    5.75  .25   8.0     3.5
1.96    5.5   .25   9.0     1.5
2.08    5.2   .25   11.0    1.0
2.2     5.0   .25   12.0    0.5
2.3     5.0   .25   13.0    0.1
2.4     4.5   .25   14.0    0.07
2.5     4.0   .25   15.0    0.11
2.6     3.5   .25   16.0    0.02
