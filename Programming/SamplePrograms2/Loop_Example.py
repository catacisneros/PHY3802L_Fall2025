import numpy as np
import LT.box as B

newx=np.empty(20)
newy=np.empty(20)
newdy=np.empty(20)
for array_index in range (20):
    newx [array_index]=array_index
    newy [array_index]=array_index**2
    newdy [array_index]=np.sqrt(array_index)

B.pl.figure()
B.plot_exp (newx, newy,dy=newdy)