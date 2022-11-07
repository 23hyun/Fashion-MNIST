from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

import pandas as pd
import numpy as np

INPUT_SHAPE = 784
NUM_CATEGORIES = 10

LABEL_DICT = {
 0: "T-shirt/top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

# 데이터 불러오기
train_raw = pd.read_csv('C:/Users/user/Desktop/sehyun/archive/fashion-mnist_train.csv').values
test_raw = pd.read_csv('C:/Users/user/Desktop/sehyun/archive/fashion-mnist_test.csv').values

# 원-핫 인코딩 후에 x와 y로 분리
train_x, train_y = (train_raw[:,1:], to_categorical(train_raw[:,0], num_classes = NUM_CATEGORIES))
test_x, test_y = (test_raw[:,1:], to_categorical(test_raw[:,0], num_classes = NUM_CATEGORIES))

# x데이터 정규화하기
train_x = train_x / 255
test_x = test_x / 255

# 모델 구축하기
model = Sequential()

model.add(Dense(512, input_dim = INPUT_SHAPE))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(NUM_CATEGORIES))
model.add(Activation('softmax'))

# 다중 선택 분류 - 범주형 교차 엔트로피
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련하기
model.fit(train_x,
          train_y,
          epochs = 10,
          batch_size = 32,
          validation_data = (test_x, test_y))

#수치 나타내기
score = model.evaluate(train_x, train_y, steps=math.ceil(10000/32))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
