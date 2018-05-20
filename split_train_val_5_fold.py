import os
import numpy as np
import shutil

# 为5折交叉验证做准备
# 生成的数据分别存放在：train_split1, val_split1;  train_split2,val_split2;  train_split3, val_split3;
#                       train_split4, val_split4; train_split5, val_split5 这些文件夹中
np.random.seed(2016)

n_fold = 5   # 5 折交叉验证

root_total = r'./data/train'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

for fish in FishNames:
    nbr_train_samples = 0
    nbr_val_samples = 0
    total_images = os.listdir(os.path.join(root_total, fish))
    np.random.shuffle(total_images)  # 扰乱total_images中元素的顺序
    for k in range(1, n_fold + 1):  # k 取值为 1， 2， 3， 4， 5
        root_train = r'./data/train_split' + str(k)
        root_val = r'./data/val_split' + str(k)
        nbr_train_samples = 0
        nbr_val_samples = 0
        if not os.path.exists(root_train):
            os.mkdir(root_train)
        if not os.path.exists(root_val):
            os.mkdir(root_val)
        if fish not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, fish))   # 不用自己添加 / ,看来os.path.join很好用

        each_fold_size = int(len(total_images)/5)

        val_images = total_images[each_fold_size * (k-1):each_fold_size * k]
        train_images = list(set(total_images).difference(set(val_images)))

        for img in train_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_train, fish, img)
            shutil.copy(source, target)
            nbr_train_samples += 1

        if fish not in os.listdir(root_val):
            os.mkdir(os.path.join(root_val, fish))

        for img in val_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_val, fish, img)
            shutil.copy(source, target)
            nbr_val_samples += 1
        print('{} {} th part finished!'.format(fish, k))
        print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
    print('{} total number is {}'.format(fish, nbr_train_samples + nbr_val_samples))
    print('{} finished!'.format(fish))
print('Finish splitting train and val images!')


