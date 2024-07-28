import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

look_back = 45
cols = list(range(look_back))

dataframe = pd.read_csv('derivadas_alt_15.csv', usecols=cols, engine= "python")
dataset = dataframe.values
dataset = dataset.astype("float32")

def criardataset(dataset, defasagem):
    dataX, dataY = [], []
    for i in range(len(dataset)-defasagem-11):
        x = dataset[i, 0:45]
        if(len(x) !=45):print(i)
        dataX.append(x)
        y = dataset[i+defasagem:i+defasagem+12, 2]
        if(len(y)!=12):print(i)
        dataY.append(y)
    
    return np.array(dataX), np.array(dataY)

def create_dataset(dataset,look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), ]
        dataX.append(a)
        dataY.append(dataset[i+look_back: i+look_back+12, 2])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print(dataX.shape)
    print(dataY.shape)
    return np.array(dataX), np.array(dataY)

def erro_percentual_medio_absoluto(previsto, real):
    ape = []
    for i in range(min(len(previsto),len(real))):
        per_err = ((real[i][0]-previsto[i][0])/real[i][0])
        per_err = abs(per_err)
        ape.append(per_err)

    mape = sum(ape)/len(ape)
    return mape


train_size = int(len(dataset)*0.7)
test_size = int(len(dataset)*0.15)
validation_size = len(dataset) - (train_size+test_size)
train, test, validation = dataset[:train_size, :], dataset[train_size:train_size+test_size, :], dataset[-validation_size: , :]

trainX, trainY = criardataset(train, 15)
print(trainX.shape)
print(trainY.shape)

testX, testY = criardataset(test, 15)
validationX, validationY = criardataset(validation, 15)

model = Sequential()
model.add(Dense(units=45, input_shape=(45,),activation='relu'))
model.add(Dense(2))
model.add(Dense(12))

model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(trainX, trainY, epochs=10, batch_size = 1, verbose = 1, validation_data=(validationX, validationY))

trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

model.save("models/modeloderivadas15.keras")

model = keras.models.load_model("modeloderivadas15.keras")


testPredict = model.predict(testX)

erromedio = erro_percentual_medio_absoluto(testPredict,testY)
print(erromedio)
#plt.plot(dataset[ train_size:-validation_size , :1])
plt.plot(testY[: , :1])
plt.plot(testPredict[: , :1])
plt.text(-100,40000, "MAPE:" +str(erromedio))
plt.legend("T")
plt.show()