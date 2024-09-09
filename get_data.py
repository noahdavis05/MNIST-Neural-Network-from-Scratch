import tensorflow as tf
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

save_dir = './mnist/'

os.makedirs(save_dir, exist_ok=True)

import numpy as np

np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

print("datasets saved!!")