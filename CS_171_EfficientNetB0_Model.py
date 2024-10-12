import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# We used the pretrained model and used some preprocessing and
# added layers to the end of the EfficientNetB0 model to
# make it run.

# file path from archive folder w/ bird images
train_set_path = r"archive/train"
test_set_path = r"archive/test"
valid_set_path = r"archive/valid"


# Using preprocessing will convert images into Dataset type,
# categorical seems to set one hot encoding, so we don't need to implement it
# Using original image size of 224 x 224
#

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_set_path,  # directory
    subset="training",
    label_mode="categorical",  # does not need one hot encoding i guess?
    image_size=(224, 224),  # set size of images
    validation_split=0.2,
    shuffle=True,
    seed=33,
    labels="inferred",  # gets label from subdirectories (should be the names of birds)
    batch_size=32
)
#
validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    valid_set_path,  # directory
    label_mode="categorical",
    image_size=(224, 224),  # set size of images
    labels="inferred",  # gets label from subdirectories (should be the names of birds)

)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_set_path,  # directory
    label_mode="categorical",
    image_size=(224, 224),  # set size of images
    labels="inferred",  # gets label from subdirectories (should be the names of birds)
)

inputs = tf.keras.Input(shape=(224, 224, 3)) # for input tensor of the eff model

# Our pretrained model we are using
eff_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False,  # used to set tensors shape
    input_shape=(224, 224, 3),  # sets image shape expected
    input_tensor = inputs,
    classes=525
)

eff_model.trainable = False

# The layer added to the EfficientNetB0 model
# used higher strides, 512 filters
convolution = tf.keras.layers.Conv2D(filters=512, activation="relu", strides=(3, 3), kernel_size=3)(eff_model.output)
pooling = layers.GlobalAveragePooling2D()(convolution)
output_layer = layers.Dense(525, activation="softmax")(pooling)

eff_model = tf.keras.Model(inputs, output_layer)  # needed to shape tensors

# uses the default value for adam learning
eff_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy']
                  )

eff_model.summary()

efficient_net_history = eff_model.fit(train_data, epochs=20, validation_data=validation_data
                                      )
# For our metrics
accuracy = efficient_net_history.history["accuracy"]
validation_accuracy = efficient_net_history.history["val_accuracy"]
loss = efficient_net_history.history["loss"]
validation_loss = efficient_net_history.history["val_loss"]
x_range = range(1, len(loss) + 1)

# Plotting the graphs for loss and accuracy
plt.plot(x_range, accuracy)
plt.plot(x_range, validation_accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Training and Valid Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.figure()

plt.plot(loss)
plt.plot(validation_loss)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Training and Valid Loss')
plt.legend(['Training Loss', 'Validation Loss'])

plt.show()

eff_model.evaluate(test_data)

