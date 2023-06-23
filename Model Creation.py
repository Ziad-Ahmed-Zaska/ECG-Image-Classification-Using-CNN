from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#CNN architecture creation with three convolutional layers, two max-pooling layers, and two fully connected layers
classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(224, 224, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(6, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

#Data augmentation and preprocessing for the training set using ImageDataGenerator.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#Preprocessing for the test set using only rescaling.
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Loading the training set using flow_from_directory method.
training_set = train_datagen.flow_from_directory('archive/ECG Heartbeat Categorization Dataset Image Version/train',
                                                 target_size=(224, 224),
                                                 color_mode="grayscale",
                                                 batch_size=32,
                                                 class_mode='categorical'
                                                 )

#Loading the test set using flow_from_directory method.
test_set = test_datagen.flow_from_directory('archive/ECG Heartbeat Categorization Dataset Image Version/test',
                                            target_size=(224, 224),
                                            color_mode="grayscale",
                                            batch_size=32,
                                            class_mode='categorical'
                                            )

from keras.callbacks import EarlyStopping, ModelCheckpoint

#Defining two callbacks to be used during model training, EarlyStopping and ModelCheckpoint.
#EarlyStopping will stop training if the validation loss doesn't improve for 10 consecutive epochs.
#ModelCheckpoint will save the best model weights based on validation accuracy.

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),

    ModelCheckpoint(
        filepath='Output Model/CNN_model.h5',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

import os

train_dir = 'archive/ECG Heartbeat Categorization Dataset Image Version/train'

num_images_per_class = []
for class_dir in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_dir)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        num_images_per_class.append(num_images)
        print(f"Class {class_dir}: {num_images} images")

print(f"Total number of images: {sum(num_images_per_class)}")

total_train_images = sum(num_images_per_class)
batch_size = 32 # or any other batch size you want to use
steps_per_epoch = total_train_images // batch_size

#Training the CNN classifier using the fit method.
result = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    callbacks=callbacks,
    validation_data=test_set,
    validation_steps=553
)

from keras.models import load_model

#Loading the best saved model weights from the ModelCheckpoint callback.
best_model=load_model('Output Model/CNN_model.h5')
#Evaluating the best model on the test set and printing the test loss and accuracy.

results = best_model.evaluate(test_set, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

import matplotlib.pyplot as plt

#Plotting the training and validation losses over epochs and saving the plot as a JPEG image.
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('Output Plots/loss.jpg', dpi=500, bbox_inches='tight')
plt.show()

#Plotting the training and validation accuracies over epochs and saving the plot as a JPEG image.
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('Output Plots/accuracy.jpg', dpi=500, bbox_inches='tight')
plt.show()
