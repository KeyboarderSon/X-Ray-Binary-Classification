import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import densenet
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
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
"""
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
"""
train_datagen = ImageDataGenerator(samplewise_center=True,
                                samplewise_std_normalization=True,
                                height_shift_range=0.05,
                                rotation_range=5,                   
                                shear_range = 0.1,
                                zoom_range = 0.15,
                                horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('../origin_split/train',
                                                 target_size = (256, 256),
                                                 batch_size = batch_size,
                                                 color_mode = 'rgb',
                                                 class_mode = 'binary')

val_set = test_datagen.flow_from_directory('../origin_split/val',
                                            target_size = (256, 256),
                                            batch_size = 16,
                                            color_mode = 'rgb',
                                            class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../origin_split/val',
                                            target_size = (256, 256),
                                            batch_size = 1,
                                            color_mode = 'rgb',
                                            class_mode = 'binary')


if not TEST_MODEL:
    densenet121 = densenet.DenseNet121(weights=None, input_shape=(256, 256, 3))


    #model=load_model('my_model.h5')

    model = Sequential([
        #densenet121,
        load_model('my_model.h5'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])


    model.summary()
    #model.load_weights('my_model.h5')

    #로그 파일이 어디에서 남겨지지?
    tensorboard = TensorBoard(log_dir=".\logs")

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


    history = model.fit_generator(train_set,
                         steps_per_epoch = len(train_set),
                         epochs = epochs,
                         validation_data = val_set,
                         callbacks=[learning_rate_reduction, early_stop, checkpoint, tensorboard]
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
test_score = model.evaluate_generator(test_set, verbose=2)
print('\nModel Accuracy: ', test_score[1])

print('\nParameters used:',
	'\ntrain_samples:   ', len(train_set),
	'\nepochs:          ', epochs,
	'\nbatch_size:      ', batch_size,
	'\ninit_learn_rate: ', lr)
