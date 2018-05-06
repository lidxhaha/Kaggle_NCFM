from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 25
batch_size = 10
train_data_dir = r'./data/train_split'
val_data_dir = r'./data/val_split'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
# InceptionV3_model.summary()

optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# autosave best Model
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='acc', verbose=1, save_best_only=True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,  # 将图片中每个像素都乘以1./255
        shear_range=0.1,  # 浮点数，剪切强度（逆时针方向的剪切变换角度），取0.1实测没什么效果
        zoom_range=0.1,  # 浮点数或形如[lower, upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower, upper] = [1-zoom_range,1+zoom_range]
        rotation_range=10.,   # 数据提升时图片随机转动的角度，是一个0-180的度数，取值为0-180
        width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度，取值0-1
        height_shift_range=0.1,  # 同上，图片随机竖直偏移的幅度
        horizontal_flip=True)  # 随机的对图片进行水平翻转，实测结果显示就是以竖直线为轴，进行轴对称

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),  # 此处实现对不同尺寸的原始图片进行尺寸变化，来使输入的尺寸统一。
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes=FishNames,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        # save_prefix = 'aug',
        classes=FishNames,
        class_mode='categorical')

"""
InceptionV3_model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        callbacks=[best_model])
#  经历一次epoch之后会调用一次callback
"""

# 新版本的keras应该这样写：
InceptionV3_model.fit_generator(
        train_generator,
       #  steps_per_epoch=nbr_train_samples/batch_size,
        steps_per_epoch=5,
        epochs=nbr_epochs,
        validation_data=validation_generator,
        validation_steps=nbr_validation_samples/batch_size,
        callbacks=[best_model]   # 每进行完一个epoch保存一下模型
)

