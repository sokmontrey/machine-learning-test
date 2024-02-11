import numpy as np
import keras
import matplotlib.pyplot as plt

raw_train = np.genfromtxt('./dataset/train.csv', delimiter=',', skip_header=1, dtype=str)
raw_test = np.genfromtxt('./dataset/test.csv', delimiter=',', skip_header=1, dtype=str)

input_train = raw_train[:, [2, 4, 5, 6, 7, 9]]
input_train[input_train == ""] = 0
input_train[:, 1] = (raw_train[:, 1] == "female").astype(float)
input_train = input_train.astype(float)
label_train = raw_train[:, 1].astype(float)

input_test = input_train[750:]
label_test = label_train[750:]

input_train = input_train[:750]
label_train = label_train[:750]

input_submi = raw_test[:, [1, 3, 4, 5, 6, 8]]
input_submi[input_submi == ""] = 0
input_submi[:, 1] = (raw_test[:, 1] == "female").astype(float)
input_submi = input_submi.astype(float)
input_submi = input_submi.astype(float)

print(input_train.shape)
print(label_train.shape)
print(input_submi.shape)

model = keras.Sequential([
    keras.layers.Dense(units=5, input_shape=[6], activation='sigmoid'),
    keras.layers.Dense(units=3, activation='sigmoid'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(input_train, label_train, epochs=500, batch_size=16, 
                    verbose=True, validation_data=(input_test, label_test))

print("Finished training the model")

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

prediction = model.predict(input_submi)
prediction_str = ""
for i in prediction.reshape(418,):
    if i > 0.5:
        prediction_str += '1\n'
    else:
        prediction_str += '0\n'

with open('./result.txt', 'w') as f:
    f.write(prediction_str)
    f.close()
