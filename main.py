import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fix git

def main():
    celc = np.array([32, -60, 86, -47, -273, 4, 30, 23, -34, 11])

    farh = np.array([89.6,-76,186.8,-52.6,-459.4,39.2,86,73.4,-29.2,51.8])

    layer0 = tf.keras.layers.Dense(units=3, input_shape=[1])
    layer1 = tf.keras.layers.Dense(units=2)
    layer2 = tf.keras.layers.Dense(units=1)

    model = tf.keras.Sequential([layer0, layer1, layer2])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

    history = model.fit(celc, farh, epochs=1000, verbose=False)

    print("Finished training the model")

    print(model.predict([17, 67, 82, 54]))

    print(f'Layer variable: {layer0.get_weights()}')

if __name__ == "__main__":
    main()
