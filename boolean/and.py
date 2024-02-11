import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [0], [0], [1]])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[2], activation='sigmoid')
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inputs[:, 0], inputs[:, 1], outputs, c='r', marker='o')

    model.compile(optimizer=tf.keras.optimizers.Adam(0.3)
                  , loss='binary_crossentropy', metrics=['accuracy'])

    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = model.get_weights()[0][0] * X + model.get_weights()[0][1] * Y + model.get_weights()[1][0]
    # ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color='r')

    history = model.fit(inputs, outputs, epochs=500, verbose=True)

    print("Finished training the model")

    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = model.get_weights()[0][0] * X + model.get_weights()[0][1] * Y + model.get_weights()[1][0]
    # ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color='b')

    plt.show()


if __name__ == "__main__":
    main()
