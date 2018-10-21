from keras.engine import Model
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
import cv2
import pickle
import os.path
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import config
from helpers import resize_to_fit
from keras.optimizers import Adam

directory_path = os.path.expanduser(config.directory_path)
MODEL_LABELS_FILENAME = config.MODEL_LABELS_FILENAME
MODEL_FILENAME = config.MODEL_FILENAME

# initialize the data and labels
data = []
labels = []

epochs = config.EPOCHS
image_size = config.IMAGE_SIZE
batch_size = config.BATCH_SIZE

# loop over the input images

print('looping over input images')

for root, dirs, files in os.walk(directory_path):
    for filename in files:

        file_name_no_extension, file_extension = os.path.splitext(filename)

        if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
            full_path = os.path.join(root, filename)

            image = cv2.imread(full_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the letter so it fits in a 20x20 pixel box
            #image = resize_to_fit(image, 20, 20)
            image = resize_to_fit(image, image_size, image_size)

            #image = np.expand_dims(image, axis=2)

            #print(image.shape)

            # Add a third channel dimension to the image to make Keras happy


            label = int(os.path.basename(root))

            # Add the letter image and it's label to our training data
            data.append(image)
            labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print('splitting data')

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.3, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Convolution Features

hidden_dim = 512
nb_class = 17

vgg_model = VGGFace(include_top=False, input_shape=(image_size, image_size, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)


# After this point you can use your model to predict.
# ...

custom_vgg_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# Train the neural network
custom_vgg_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs, verbose=1)

# Save the trained model to disk
custom_vgg_model.save(MODEL_FILENAME)

print('done')