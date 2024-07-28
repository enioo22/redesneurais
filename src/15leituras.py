import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

cols = list(range(15))

look_back = len(cols)

dataframe = pd.read_csv('apenas_leituras_15.csv', usecols=cols, engine= "python")
dataset = dataframe.values
dataset = dataset.astype("float32")


def create_dataset(dataset,look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0:12])
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

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
validationX, validationY = create_dataset(validation, look_back)


model = Sequential()
model.add(Dense(units=15, input_shape=(look_back,),activation='relu'))
model.add(Dense(2))
model.add(Dense(12))

model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(trainX, trainY, epochs=12, batch_size = 1, verbose = 1)

trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

model.save("models/modelo15.keras")

model = keras.models.load_model("models/modelo15.keras")

testPredict = model.predict(testX)

erromedio = erro_percentual_medio_absoluto(testPredict,testY)

print(erromedio)

#plt.plot(dataset[ train_size:-validation_size , :1])
plt.plot(testY[: , :1])
plt.plot(testPredict[: , :1])
plt.text(-100,40000, "MAPE:" +str(erromedio))
plt.legend("T")
plt.show()