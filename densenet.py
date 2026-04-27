import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(tf.config.list_physical_devices('GPU'))
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '140k_dataset/real_vs_fake/real-vs-fake/train',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary'
)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '140k_dataset/real_vs_fake/real-vs-fake/valid',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '140k_dataset/real_vs_fake/real-vs-fake/test',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary',
    shuffle=False) 

print(train_ds.class_names)

#Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    ])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (augmentation(preprocess_input(x), training=True),y), num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (preprocess_input(x),y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x),y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_densenet121.keras', monitor='val_loss', save_best_only=True)

base = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
    name='DenseNet121'
)
base.trainable = True

inputs = tf.keras.layers.Input(shape=(256, 256, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_ds, validation_data=valid_ds, epochs=10, callbacks=[early_stopping, checkpoint])

best_model = tf.keras.models.load_model('best_densenet121.keras')
y_labels = np.concatenate([y.numpy().astype(int).ravel() for _, y in test_ds])
y_prob = best_model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = float(accuracy_score(y_labels, y_pred))
prec = float(precision_score(y_labels, y_pred, average='binary', zero_division=0))
rec = float(recall_score(y_labels, y_pred, average='binary', zero_division=0))
f1 = float(f1_score(y_labels, y_pred, average='binary', zero_division=0))

#Print performance metrics
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'F1 Score: {f1}')

print('Classification Report: ', classification_report(y_labels, y_pred, zero_division=0))

#Confusion matrix
cm = confusion_matrix(y_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()