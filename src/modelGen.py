import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("../data/Shakespeare_data.csv")
del df['Dataline']
df = df.dropna()
df = df.reset_index(drop=True)
df = df.copy() # Not sure why the copy is necessary but it avoid errors
df['PlayerLinenumber'] = df['PlayerLinenumber'].apply(int)

parseAsl = lambda asl : [int(s) for s in asl.split('.')]

parsedAsl = pd.DataFrame(df['ActSceneLine'].apply(parseAsl).tolist(), columns=['A','S','L'])
df[['A','S','L']] = parsedAsl[['A','S','L']]
del df['ActSceneLine']

shuffled = df.sample(frac=1)

oneHot = shuffled.join(pd.get_dummies(shuffled['Play']))
labels = pd.get_dummies(shuffled['Player'])
del oneHot['Play']
del oneHot['Player']

noText = oneHot.copy()
del noText['PlayerLine']

lenTotal = len(noText)
lenTrain = int(.9*lenTotal)
lenTest  = lenTotal - lenTrain

trainingInput  = noText.head(lenTrain).astype('float').to_numpy()
trainingLabels = labels.head(lenTrain).astype('float').to_numpy()

testingInput  = noText.head(lenTest).astype('float').to_numpy()
testingLabels = labels.head(lenTest).astype('float').to_numpy()

lenInput  = len(noText.columns)
lenLabels = len(labels.columns)
lenHidden = int((lenInput+lenLabels)/2)

model = tf.keras.Sequential([
    keras.layers.Dense(lenHidden, activation='relu'),
    keras.layers.Dense(lenLabels, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
callback = keras.callbacks.ModelCheckpoint(filepath='../models/noText.ckpt',
                                           save_weights_only=True,
                                           verbose=1)
model.fit(trainingInput, trainingLabels, epochs=100, callbacks=[callback], verbose=1)

model2 = tf.keras.Sequential([
    keras.layers.Dense(lenHidden, activation='relu'),
    keras.layers.Dense(lenHidden, activation='relu'),
    keras.layers.Dense(lenHidden, activation='relu'),
    keras.layers.Dense(lenLabels, activation='softmax'),
])
model2.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
callback = keras.callbacks.ModelCheckpoint(filepath='../models/noText_deep.ckpt',
                                           save_weights_only=True,
                                           verbose=1)
model2.fit(trainingInput, trainingLabels, epochs=50, callbacks=[callback], verbose=1)
