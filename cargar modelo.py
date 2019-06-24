from keras.models import Sequential
from keras.models import load_model
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
from numpy import genfromtxt

model = load_model('/home/dugo/Bases de datos TFG/Entrenamientos/1/model_epoch_10.hdf5')

testN = np.memmap("/home/dugo/Bases de datos TFG/positive_test", dtype='float32', mode='r', shape=(3369*60, 80))
X_test = np.reshape(testN, (int(len(testN)/60), 60, 80, 1))
plt.imshow(testN[0:60*10, :])
plt.show()
testN = np.reshape(testN, (int(len(testN)/60), 60, 80))

results = model.predict(X_test)

percentage = 0
print(results.size)
print(testN.shape)

for x in range(0,results.size):
    if results[x] > 0.99:
        percentage = percentage + 1
        img = testN[x, :, :]

        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

print(percentage/results.size)
