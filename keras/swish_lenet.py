import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn import datasets
from sklearn.model_selection import train_test_split


def swish(x):
    return x * K.sigmoid(x)


np.random.seed(1234)

'''
Generate Model
'''
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 30000  # num of data we use
N_train = 20000
N_validation = 4000
indices = np.random.permutation(range(n))[:N]

X = mnist.data[indices]
X = X / 255.0
X = X - X.mean(axis=1).reshape(len(X), 1)
X = X.reshape(N, 28, 28, 1)
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=N_train)
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_train, Y_train, test_size=N_validation)

'''
Configure Model
'''
n_in = len(X[0])  # 784
n_out = len(Y[0])  # 10


def weight_variable(shape, name=None):
    return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()

# 1st Convolution Layer
model.add(Conv2D(32, (5, 5), padding='same',
                 data_format='channels_last',
                 input_shape=(28, 28, 1)))
model.add(Activation(swish))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution Layer
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation(swish))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation(swish))
model.add(Dropout(0.5))

# Readout Layer
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

'''
Train Model
'''
epochs = 200
batch_size = 200

hist = model.fit(X_train, Y_train, epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation),
                 callbacks=[early_stopping])

'''
Visualize Train & Validation
'''
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(val_loss)), val_loss, label='loss', color='black')
plt.xlabel('epochs')
plt.show()

'''
Validate Model
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
