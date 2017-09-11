import cv2
import numpy as np
import random

from glob import glob
from keras.utils import np_utils
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.datasets import load_files
from tqdm import tqdm


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
def cnn_model():
    model = Sequential([
        GlobalAveragePooling2D(input_shape=train_resnet.shape[1:]),

        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu'),

        BatchNormalization(),
        Dropout(0.2),
        Dense(133, activation='softmax')
        ])
    return model
# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

random.seed(0)

# load filenames in shuffled human dataset
human_files = np.array(glob("humanImages/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')




# Return accuracy of dog / human detection
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
human_acc = np.mean(list(map(face_detector, human_files_short)))
dog_acc = np.mean(list(map(dog_detector, dog_files_short)))
print('Human Accuracy:\t{:.2f}%'.format(100 * human_acc))
print('Dog Accuracy:\t{:.2f}%'.format(100 * dog_acc))

# Get Resnet bottleneck features
bottleneck_features = np.load('DogResnet50Data.npz')
train_resnet = bottleneck_features['train']
valid_resnet = bottleneck_features['valid']
test_resnet = bottleneck_features['test']




resnet_model = cnn_model()
print(resnet_model.summary())

# Compile ResNet-50 model
resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train ResNet-50 model
checkpointer = ModelCheckpoint(filepath='saved models/weights.best.Resnet50.hdf5',
                               verbose=1, save_best_only=True)

resnet_model.fit(train_resnet, train_targets, validation_data=(valid_resnet, valid_targets), epochs=20, batch_size=32,
                 callbacks=[checkpointer], verbose=1)

# Load weights
resnet_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

# Test ResNet-50 model
resnet_predictions = [np.argmax(resnet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_resnet]
test_accuracy = 100*np.sum(np.array(resnet_predictions) == np.argmax(test_targets, axis=1))/len(resnet_predictions)
print('Test accuracy:\t{:.2f}%'.format(test_accuracy))


