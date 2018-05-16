Kaggle_NCFM
===

1.这是kaggle的一个比赛，主要是对各种鱼进行分类，用到了CNN，fine tune以及data augmentation, kaggle链接：https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring <br>

2.重要的是这个程序提供了一种用Keras处理图像问题的框架程序，可以供参考。 <br>

3.一个重要的trick是，不仅仅将data augmentation用于训练集，而且还将其用于测试集，将同一张图片的不同变换都拿来测试并得到softmax的输出结果，然后再进行平均。 <br>
  
4.本程序的数据集和训练好的模型：链接：https://pan.baidu.com/s/1a2Rn8tejGDxgoNrI_Fxoow 密码：8spa  <br>
>>网盘文件介绍：　 <br>
>>data.rar: 数据集 <br>
>>weights_1.h5 : 25个epoch后的model <br>
>>weights_2.h5 : 50个epoch后的model <br>
\<br>
5. 文件介绍：<br>
>>predict.py : 不用average augmentation的在测试集上的预测 <br>
>>predict_average_augmentation.py : 使用了average augmentation的在测试集上的预测 <br>
>>split_train_val.py : 划分测试集和训练集 <br>
>>train.py : 训练模型的程序 <br>
>>submit_1_no_predict_augmentation.csv : 跑25个epoch并且不用aveage augmentation的提交文件，分数是1.08447 <br>
>>submit_1_with_predict_augmentation.csv : 跑25个epoch并且使用aveage augmentation的提交文件，分数是1.00316 <br>
>>submit_2_no_predict_augmentation.csv : 跑50个epoch并且不用aveage augmentation的提交文件，分数是1.14559 <br>
>>submit_2_with_predict_augmentation.csv : 跑50个epoch并且使用aveage augmentation的提交文件，分数是1.02733 <br>
<br>
6. 跑25个epoch时，val_acc大概在0.91,跑50个epoch时，train_loss: 0.1404, train_acc: 0.9632, val_loss: 0.2187, val_acc: 0.94459；<br>
但是在测试集上的表现，都是25个epoch的要好，看来不能太迷信val_acc,val_acc越高，不一定测试集上的精度越高。 <br>



