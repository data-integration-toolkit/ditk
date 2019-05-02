import matplotlib.pyplot as plt 

plt.plotfile('contextweighted_curve.dat', delimiter=' ', cols=(0, 1), 
             names=('Recall', 'Precision'), marker='o')
plt.show()