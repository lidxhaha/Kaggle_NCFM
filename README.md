# Kaggle_NCFM
1.这是kaggle的一个比赛，主要是对各种鱼进行分类，用到了CNN，fine tune以及data augmentation, kaggle链接：https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

2.重要的是这个程序提供了一种用Keras处理图像问题的框架程序，可以供参考。

3.一个重要的trick是，不仅仅将data augmentation用于训练集，而且还将其用于测试集，将同一张图片的不同变换都拿来测试并得到softmax的输出结果，然后再进行
  平均，这样做分数由1.08447提升到了1.00316
  
4.本程序的数据集和训练好的模型：链接：https://pan.baidu.com/s/1a2Rn8tejGDxgoNrI_Fxoow 密码：8spa


----------------- 我是分隔线 -----------------------------------------
1. 之前跑完25个epoch后，在验证集上的精度达到了0.91，于是我愚蠢的停止了训练，认为这么差的数据能达到0.91已经太不容易了。

2. 但是我错了，再次跑数据，这次：train_loss: 0.1404, train_acc: 0.9632, val_loss: 0.2187, val_acc: 0.94459,结果显示这次真的很难再提升了。
