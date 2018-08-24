import numpy as  np
from random import randint
from keras.models import load_model


def one_hot_encode(X, len, num):
    x = np.zeros((batch_size, len, num), dtype=np.float32)
    for i, batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[i, j, elem] = 1
    return x


model = load_model("deepsort.h5")

batch_size = 32
len = 100
num = 100
r = randint(2, len)
testX = np.random.randint(num, size=(1, r))
print("Test:",testX)
for i in range(len - r):
      testX = np.append(testX, [0])

testX = testX.reshape((1, len))
#print(testX)
test = one_hot_encode(testX, len, num)
y = model.predict(test, batch_size=1)
rnn_sorted = np.argmax(y, axis=2)[0]
for i in range(len):
     if rnn_sorted[0] == 0:
         rnn_sorted = np.delete(rnn_sorted, 0)

print("Result:",rnn_sorted)