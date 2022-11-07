```python
import os
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import visualkeras
from keras.utils import plot_model
import math
from keras.optimizers import RMSprop

train_dataset = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_dataset = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

def data_preprocessing(raw):
    label = tf.keras.utils.to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    image = x_shaped_array / 255
    return image, label

X, y = data_preprocessing(train_dataset)
X_test, y_test = data_preprocessing(test_dataset)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

model = tf.keras.Sequential()

# First layer, which has a 2D Convolutional layer with kernel size as 3x3 and Max pooling operation 
model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second layer, which has a 2D Convolutional layer with kernel size as 3x3 & ReLU activation and Max pooling operation 
model.add(Conv2D(64, (3,3), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Fully connected layer with ReLU activation function 
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))

# Output layer with softmax activation function
model.add(Dense(10, activation=tf.nn.softmax))

#모델 요약
model.summary()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a00bf8ea-0894-4764-a2aa-cbaa19177362/Untitled.png)

```python
#visualkeras로 레이어 모델 보기
visualkeras.layered_view(model)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d0acf0be-9ecc-4682-b70f-4b41c224d05d/Untitled.png)

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train_model = model.fit(X_train, y_train,
                  batch_size=Batch_size,
                  epochs=No_epochs,
                  verbose=1,
                  validation_data=(X_val, y_val))
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56f2b587-00dc-420f-8dd5-dfd18e2ff413/Untitled.png)

모델 튜닝하기( 배치 정규화+드롭아웃 레이어, lr 조절)

```python
model = Sequential()

model.add(Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_normal', input_shape=(28,28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
```

```python
# 튜닝 후 visualkeras로 레이어 모델 보기
visualkeras.layered_view(model)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/537b0a72-140c-49a7-a006-9184c54f32c6/Untitled.png)

```python
#아담 옵티마이저 학습률 조정(정확률을 높이기 위해서)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = optimizer,
              loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

train_model = model.fit(X_train, y_train,
                  batch_size=Batch_size,
                  epochs=No_epochs,
                  verbose=1,
                  validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, steps=math.ceil(10000/32))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bc7aef26-e742-4186-89f1-f71fb4e9fd5b/Untitled.png)

```
#fashion mnist 사진으로 나타내기

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, axin enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title = f"Real Class is{labels[y_test[i].argmax()]}\nPredict Class is{labels[y_pred[i].argmax()]}");
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d2e8ff43-c354-4f1b-9b3a-757ef71b0820/Untitled.png)
