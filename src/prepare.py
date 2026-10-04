import os
import numpy as np
import tensorflow as tf
def main():
    os.makedirs(os.path.join("data","raw"), exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() 
    np.savez_compressed(os.path.join("data","raw","train.npz"), image=x_train, label=y_train)
    np.savez_compressed(os.path.join("data","raw","test.npz"), image=x_test, label=y_test)
if __name__ == "__main__":
    main()
