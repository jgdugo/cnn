import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import misc
from numpy import genfromtxt




trainN = genfromtxt('/home/dugo/Bases de datos TFG/negative_depth', delimiter='\t', max_rows=60*100)
X_train = trainN[:,0:80]

X_train = np.array(X_train)
np.save("/home/dugo/Bases de datos TFG/negative_depth_reduced", X_train)

trainN = genfromtxt('/home/dugo/Bases de datos TFG/positive_depth', delimiter='\t', max_rows=60*100)
X_train = trainN[:,0:80]

X_train = np.array(X_train)
np.save("/home/dugo/Bases de datos TFG/positive_depth_reduced", X_train)

trainN = genfromtxt('/home/dugo/Bases de datos TFG/negative_depth_09_06', delimiter='\t', max_rows=60*100)
X_train = trainN[:,0:80]

X_train = np.array(X_train)
np.save("/home/dugo/Bases de datos TFG/negative_depth_09_06_reduced", X_train)

trainN = genfromtxt('/home/dugo/Bases de datos TFG/positive_depth_09_06', delimiter='\t', max_rows=60*100)
X_train = trainN[:,0:80]

X_train = np.array(X_train)
np.save("/home/dugo/Bases de datos TFG/positive_depth_09_06_reduced", X_train)
img = 98
X_train *= 25.5
X_train = np.round(X_train)


X_train = np.reshape(X_train, (int(len(X_train)/60), 60, 80))
print(X_train.shape)

#X_train = X_train[(60*img):(60*img+60), :]


# misc.imsave('caca.png', X_train)

# plt.imshow(X_train, cmap=plt.cm.gray)
# plt.show()



# a = np.zeros((7, 7), dtype=np.int)
# a[1:6, 2:5] = 255


'''
misc.imsave('caca.png', a)

plt.imshow(a, cmap=plt.cm.gray)
plt.show()
'''


