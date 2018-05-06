# Kaggle_NCFM
1.这是kaggle的一个比赛，主要是对各种鱼进行分类，用到了CNN，fine tune以及data augmentation,kaggle链接https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

2.重要的是这个程序提供了一种用Keras处理图像问题的框架程序，可以供参考。

3.一个重要的trick是，不仅仅将data augmentation用于训练集，而且还将其用于测试集，将同一张图片的不同变换都拿来测试并得到softmax的输出结果，然后再进行
  平均，这样做分数由1.08447提升到了1.00316
