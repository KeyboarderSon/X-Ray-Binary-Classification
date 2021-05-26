import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import Densenet121
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
from glob import glob
import cv2

TEST_MODEL = False
epochs = 50
batch_size = 32
lr = 0.0008
os.environ["CUDA_VISIBLE_DEVICES"]="2"

images_path = {}
images_path["normal"] = glob("tf/joohye/origin/normal/*.jpg")
images_path["abnormal"] = glob("tf/joohye/origin/abnormal/*.jpg")

images_class = {
    "normal" : 0,
    "abnormal" : 1
}

X=[]
Y=[]
for label in images_path:
    for image_path in image_path[label]:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        X.append(image)
        Y.append(images_class[label])

train_data, test_x, train_label, test_y = train_test_split(X, Y, random_state=66, test_size=0.3)
test_data, val_data, test_label, val_label = train_test_split(test_x, test_y, random_state=66, test_size=0.5)


if not TEST_MODEL:
    densenet = Densenet121(include_top=False, weights='my_model.h5', input_tensor=None, input_shape=(256, 256, 3))



    model = Sequential([
        densenet,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer=optimizers.SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics='acc')
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
	                                            patience=15,
	                                            verbose=1,
	                                            factor=0.5,
	                                            min_lr=0.00000001)
    
    early_stop = EarlyStopping(monitor="val_loss",
	                           mode="min",
	                           patience=30)
        
    checkpoint = ModelCheckpoint('binary.hdf5',
	                             monitor='val_loss',
	                             verbose=1,
	                             save_best_only=True,
	                             mode='min',
	                             save_weights_only=True
								)

    history = model.fit(train_data,
            validation_data=val_data,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[checkpoint]
            )

    plt.figure(1)
    plt.plot(history.history['loss'], label="TrainLoss")
    plt.plot(history.history['val_loss'], label="ValLoss")
    plt.legend(loc='best', shadow=True)
    plt.show()
    plt.savefig('Loss.png')

	#fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    plt.figure(2)
    plt.plot(history.history['acc'], label="TrainAcc")
    plt.plot(history.history['val_acc'], label="ValAcc")
    plt.legend(loc='best', shadow=True)
    plt.show()
    plt.savefig('Acc.png')

print('##### Evaluating Model on Test Data #####')
################################# Evaluate model on Test Data ############################
test_score = model.evaluate(test_data, verbose=2)
print('\nModel Accuracy: ', test_score[1])

print('\nParameters used:',
	'\ntrain_samples:   ', len(train_data),
	'\nepochs:          ', epochs,
	'\nbatch_size:      ', batch_size,
	'\ninit_learn_rate: ', lr)