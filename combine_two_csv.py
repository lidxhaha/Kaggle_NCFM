
# 将stg1和stg2的csv文件合并到一起用于提交
# 两个csv文件合并到一起时，要注意test_stg1中的图片名就是图片名，比如image00001.jpg,
# test_stg2中的图片名应该是 test_stg2/image00001.jpg 这种格式的

import pandas as pd
root_path = r'E:/pycharm_dir/Kaggle_NCFM-master/predicted_csv/'
filepath1 = root_path + r'submit_test_stg1.csv'
filepath2 = root_path + r'submit_test_stg2.csv'
stg1_dfm = pd.read_csv(filepath1)
stg2_dfm = pd.read_csv(filepath2)

str_before = 'test_stg2/'
for i in range(len(stg2_dfm)):
    stg2_dfm.iat[i, 0] = str_before + stg2_dfm.iat[i, 0]

submit = stg1_dfm.append(stg2_dfm)
submit.to_csv(root_path + r'submit.csv', index=False)



