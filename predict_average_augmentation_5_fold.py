from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 以下两项需要修改
for nbr_test_samples, stg_1_2 in [(1000, 'test_stg1'), (12153, 'test_stg2')]:
    # nbr_test_samples = 1000  # 1000 or 12153
    # stg_1_2 = 'test_stg1'   # test_stg1 or test_stg2

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    img_width = 299
    img_height = 299
    batch_size = 32

    nbr_augmentation = 5

    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    n_fold = 5
    # 将5个训练好的model都加载到InceptionV3_model这个list中
    InceptionV3_model = list()
    for k in range(1, n_fold+1):
        weights_path = 'weights' + str(k) + '.h5'
        print('Loading model {} and weights from training process ...'.format(k))
        InceptionV3_model.append(load_model(weights_path))

    test_data_dir = 'predict/' + stg_1_2

    # test data generator for prediction
    #  average augmentation
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    predictions_list = list()
    for k in range(1, n_fold+1):
        for idx in range(nbr_augmentation):
            print('{}th augmentation for testing ...'.format(idx))

            # 这是打乱数据和进行随机变换时的随机数的种子，取100000这么大范围可以大概率保证每次augmentation的结果不会一样
            random_seed = np.random.random_integers(0, 100000)  # 能取到的范围[0, 100000]

            test_generator = test_datagen.flow_from_directory(
                    test_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    shuffle=False,    # Important !!!  这样保证后面写csv文件的时候，名称和预测值不会乱
                    seed=random_seed,  # 这是打乱数据和进行随机变换时的随机数的种子，保证每次augmentation的结果不会一样
                    classes=None,
                    class_mode=None)

            test_image_list = test_generator.filenames
            print('Begin to predict for testing data ...')
            if idx == 0:
                predictions_list.append(InceptionV3_model[k-1].predict_generator(test_generator, nbr_test_samples/batch_size))
            else:
                predictions_list[k-1] += InceptionV3_model[k-1].predict_generator(test_generator, nbr_test_samples/batch_size)

        predictions_list[k-1] /= nbr_augmentation
    predictions = np.zeros(shape=predictions_list[0].shape, dtype=np.float64)
    for k in range(n_fold):
        predictions += predictions_list[k]
    predictions /= n_fold
    print('Begin to write submission file ..')
    f_submit = open('submit_' + stg_1_2 + '.csv', 'w')
    f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

    f_submit.close()

    print('Submission file successfully generated!')
