import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2, input_shape=[2], activation='sigmoid'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inputs[:, 0], inputs[:, 1], outputs, c='r', marker='o')

    plt.show()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1)
                  , loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(inputs, outputs, epochs=200, verbose=False)

    predictions = model.predict(inputs)
    print(predictions)
    print(predictions > 0.5)

if __name__ == "__main__":
    main()
