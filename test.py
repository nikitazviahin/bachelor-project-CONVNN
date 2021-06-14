import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from PIL import Image
import easygui

new_model = tf.keras.models.load_model('saved_model/my_model3_aggresive_augmentation_many_epochs_more_dropout')
new_model.summary()

def vvid():
  msg = 'Введіть шлях до зображення'
  title = 'Введення шляху до зображення'
  fieldValues2 = easygui.enterbox(msg, title)
  path_string = fieldValues2
  return path_string

image_path = vvid()

data_dir = "dataset-resized"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 96 
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names

bottle_path = image_path
bottle_path = pathlib.Path(bottle_path)

img = keras.preprocessing.image.load_img(
    bottle_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

result_string = "Це зображення належить до классу '{}' з ймовірністю {:.2f} %.".format(class_names[np.argmax(score)], 100 * np.max(score))

img = Image.open(image_path)
fig, ax = plt.subplots()
ax.imshow(img)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
fig.suptitle(result_string, fontsize=14, fontweight='bold')
plt.show()

