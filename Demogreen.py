import cv2
import keras.losses
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
import tensorflow as tf
import tensorflow.python.keras.losses
import keras
import keras.metrics
from sklearn.metrics import confusion_matrix, classification_report
from keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import sklearn.metrics as metrics
from keras.utils.np_utils import to_categorical
##############################################


learning_rate = 0.1  # Tỉ lệ học được
min_learning_rate = 0.00001  # Mức tỷ lệ học khi đạt đến giá trị này sẽ không giảm nữa
learning_rate_reduction_factor = 0.5  # Được sử dung khi giảm learning_rate -> learning_rate *= learning_rate_reduction_factor
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 1  # Điều khiển dữ liệu trả về trong quá trình train và test 0- không có ,1- báo cáo số liệu sau mỗi batch, 2- báo cáo số liệu sau mỗi epoch
image_size = (100, 100)  # kích thước hình truyền vào
input_shape = (100, 100, 3)  # Hình đầu vào cho các model, vì hình ảnh trong dataset là hình RGB 100x100

use_label_file = False  # set this to true if you want load the label names from a file; uses the label_file defined below; the file should contain the names of the used labels, each label on a separate line
label_file = 'labels.txt'
base_dir = 'input/'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test_Green') #(Apple golden: 192 , apple red: 198, Banana: 198, Banana red: 198, Lemon: 198) = 978
train_dir = os.path.join(base_dir, 'Train_Green') # (Apple golden: 448 , apple red: 461, Banana:458 , Banana red: 458, Lemon: 458) =2283
output_dir = 'output_files'  # Thư mục trả về model, dưới dạng output_files/model_name
##############################################

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]
else:
    labels = os.listdir(train_dir)
num_classes = len(labels)

# Thay đổi ngẫu nhiên màu sắc và độ bão hoà tuỳ vào điều kiện ánh sáng
def augment_image(x):
    x = tf.image.random_hue(x, 0.4)
    return x

#Hàm MSE
def mean_squared_error(y_true, y_pred):
    return backend.mean(backend.square(y_pred-y_true),axis=-1)



#  - Dùng 100% hình ảnh từ thư mục train
#    Áp dụng lật ngang và lật dọc để tăng dữ liệu và tạo các batch một cách ngẫu nhiên
#    the accuracy and loss are monitored using the validation data so that the learning rate can be updated if the model hits a local optimum
#  - Test thì không có bất kì sự gia tăng dữ liệu nào
#    Sau khi train xong thì acc và loss cuối cùng sẽ được tính vào đây
def build_data_generators(train_folder, test_folder, labels=None, image_size=(100, 100), batch_size=32):
    train_datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,  # randomly flip images
        preprocessing_function=augment_image)  # augmentation is done only on the train set (and optionally validation)

    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(train_folder, target_size=image_size, class_mode='sparse',
                                                  batch_size=batch_size, shuffle=True, subset='training', classes=labels)
    test_gen = test_datagen.flow_from_directory(test_folder, target_size=image_size, class_mode='sparse',
                                                batch_size=batch_size, shuffle=False, subset=None, classes=labels)
    return train_gen, test_gen


# chuyển hình từ rgb sang hsv và kích thước sang 100x100x4
def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    rez = tf.concat([hsv], axis=-1)
    return rez

#VGG16 5 block(2 conv, 1 maxpooling) 3 fullyconnect
def network(input_shape, num_classes):
    img_input = Input(shape=input_shape, name='data')
    x = Lambda(convert_to_hsv_and_grayscale)(img_input)
    #block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1_1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool1')(x)
    #block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2_2')(x)
    x = Activation('relu', name='conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool2')(x)
    #block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3_3')(x)
    x = Activation('relu', name='conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool3')(x)
    #block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv4_3')(x)
    x = Activation('relu', name='conv4_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool4')(x)
    #block 5
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv5_3')(x)
    x = Activation('relu', name='conv5_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool5')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fcl1')(x)
    x = Dropout(0.2)(x)
    x = Dense(4096, activation='relu', name='fcl2')(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='relu', name='fcl3')(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax', name='predictions')(x)
    rez = Model(inputs=img_input, outputs=out)
    return rez


# Hàm này sẽ thực hiện thiết lập dữ liệu, train, test
# model bất kì có thể train, đầu vào và số lượng đầu ra tuỳ thuộc vào datasets được sử dụng, trong trường hợp này là 100x100 RGB và đầu ra là lớp
# kích thước batch được sử dung để xác định số lượng hình ảnh được truyền qua mạng cùng một lúc, số batch trên mỗi epoch dược tính (tổng số hình ảnh trong set // kích thước batch) + 1
def train_and_evaluate_model(model, name="", epochs=5, batch_size=32, verbose=verbose, useCkpt=False):
    print(model.summary())
    model_out_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if useCkpt:
        model.load_weights(model_out_dir + "/model_green.h5")

    trainGen, testGen = build_data_generators(train_dir, test_dir, labels=labels, image_size=image_size, batch_size=batch_size)

    optimizer = Adadelta(lr=learning_rate)
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc',mean_squared_error])

    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=patience, verbose=verbose,
                                                factor=learning_rate_reduction_factor, min_lr=min_learning_rate)
    save_model = ModelCheckpoint(filepath=model_out_dir + "/model_green.h5", monitor='loss', verbose=verbose,
                                 save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    history = model.fit(trainGen,
                        epochs=epochs,
                        steps_per_epoch=(trainGen.n // batch_size) + 1,
                        verbose=verbose,
                        callbacks=[learning_rate_reduction, save_model])


    model.load_weights(model_out_dir + "/model_green.h5")
    trainGen.reset()
    model.evaluate(trainGen, steps=(trainGen.n // batch_size) + 1, verbose=verbose)
    model.evaluate(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)
    testGen.reset()
# Chạy code
print(labels)
print(num_classes)
model = network(input_shape=input_shape, num_classes=num_classes)

train_and_evaluate_model(model, name="fruit-360 model")
