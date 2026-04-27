import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.densenet import preprocess_input
import os

#Can load either model, just change filename
model = tf.keras.models.load_model('best_densenet121.keras')
# model = tf.keras.models.load_model('best_inceptionresnet.keras')

#Can change test image
test_image = tf.keras.preprocessing.image.load_img('140k_dataset/real_vs_fake/real-vs-fake/test/fake/0AEIDNSBKD.jpg')

actual = os.path.basename(os.path.dirname('140k_dataset/real_vs_fake/real-vs-fake/test/fake/0AEIDNSBKD.jpg'))
actual = actual.capitalize()

test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array = preprocess_input(test_image_array)

# Predict
prediction = model.predict(test_image_array, verbose=0)[0][0]
label = 'Real' if prediction >= 0.5 else 'Fake'

#Display
plt.figure(figsize=(8, 6))
plt.imshow(test_image)
plt.title(f'Prediction: {label} (Actual: {actual})')
plt.axis('off')
plt.tight_layout()
plt.show()