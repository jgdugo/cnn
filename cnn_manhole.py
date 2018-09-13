# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
from numpy import genfromtxt

# load data for training
trainN = np.memmap("negative_train", dtype='float32', mode='r', shape=(80741*60, 80))
trainP = np.memmap("positive_train", dtype='float32', mode='r', shape=(30321*60, 80))
X_train = trainN
X_train = np.append(X_train, trainP, axis=0)

X_train = np.reshape(X_train, (int(len(X_train)/60), 60, 80, 1))
Y_train = np.append(np.zeros(int(len(trainN)/60)),np.ones(int(len(trainP)/60)))
# print ("Train samples: ", len(Y_train))

# load data for testing
testN = np.memmap("negative_test", dtype='float32', mode='r', shape=(8971*60, 80))
testP = np.memmap("positive_test", dtype='float32', mode='r', shape=(3369*60, 80))
X_test = testN
X_test = np.append(X_test, testP, axis=0)
X_test = np.reshape(X_test, (int(len(X_test)/60), 60, 80, 1))
Y_test = np.append(np.zeros(int(len(testN)/60)),np.ones(int(len(testP)/60)))
# print ("Test samples: ", len(Y_test))

# create model
# Model: (5, 5, 5), max, (5, 5, 5), max, (5, 3, 3), max, (5, 3, 3), flat, 100, 1 ==> Acc: 0.9861, Val_acc:0.9578
# Model: (30, 5, 5), max, (30, 5, 5), max, (20, 3, 3), max, (10, 3, 3), flat, 150, 50, 1 ==> 
model = Sequential()
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(60, 80, 1), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(30, 5, 5, border_mode='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(20, 3, 3, border_mode='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(10, 3, 3, border_mode='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(150, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback to automatically save models on file
checkpointer = ModelCheckpoint(filepath="./model.{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)

# Fit the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=40, batch_size=100, callbacks=[checkpointer])
#history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=100)

# Save model in file
model.save("model_epoch_10.hdf5")

# Ploteo de loss y accuracy en el training y el subset de validacion
# Loss = mse
print(history.history.keys())
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend(['train','test'],loc='upper left')
plt.show()


#======================Simulacion y comparativa, errores================
#Y_pred = model.predict(X_test)
#plt.figure()
#plt.plot(Y_pred[:,0],'b')
#plt.title('Eje x')
#plt.hold(True)
#plt.plot(Y_test[:,0],'r')
#plt.title('Eje x')
#plt.legend(['Outputs','Targets'], loc= 'upper left')
#plt.xlabel("Tiempo (s)", fontsize=10)
#plt.ylabel("Velocidad angular (rad/s)", fontsize=10)
#plt.show()


