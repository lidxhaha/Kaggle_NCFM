from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

weights_path = './weights.h5'

#  这是对test_stg1进行预测，换成stg2也一样，两个csv文件合并到一起时，要注意test_stg1中的图片名就是图片名，比如image00001.jpg,
#  test_stg2中的图片名应该是 test_stg2/image00001.jpg 这种格式的
test_data_dir = './data/test_stg1/'

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)  # 只是将每个像素除以255来输入进去

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),  # 这里决定了最终图片的大小
        batch_size=batch_size,
        shuffle=False,  # Important !!! 默认是要打乱数据的，即shuffle = True
        classes=None,
        class_mode=None)

test_image_list = test_generator.filenames

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = InceptionV3_model.predict_generator(test_generator, steps= nbr_test_samples/batch_size)

np.savetxt('predictions.txt', predictions)


print('Begin to write submission file ..')
f_submit = open( 'submit.csv', 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
