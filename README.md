# Kaggle_Dog_Breed_Identification
Kaggel Dog Breed Identification
这是Kaggel的[Kaggel Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)比赛

本代码使用了mxnet的gluon最新接口。关于gluon可查看其官方文档， [https://zh.gluon.ai/](https://zh.gluon.ai/)， [http://gluon.mxnet.io/](http://gluon.mxnet.io/)

## Step
### First Step
- 从kaggle上下载数据集：
  - [train.zip](https://www.kaggle.com/c/7327/download/train.zip)
  - [test.zip](https://www.kaggle.com/c/7327/download/test.zip)
  - [labels.csv.zip](https://www.kaggle.com/c/7327/download/labels.csv.zip)
  - [sample_submission.csv.zip](https://www.kaggle.com/c/7327/download/sample_submission.csv.zip)
- 下载 [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
用于进行数据扩充

### Second Step
将下载的数据解压后， 运行 preprocessing.py 进行数据预处理

### Third Step
运行extract_features.py, 用resnet152_v1和inceptionv3预训练模型提取得到训练图片和测试集的特征向量， 并写入磁盘

```
def get_features(net, data):
    features = []
    labels = []
    for X, y in tqdm(data):
        feature = net.features(X.as_in_context(ctx))
        features.append(feature.asnumpy())
        labels.append(y.asnumpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
```

### Last Step
运行transfer_learning_train.py, 构建简单分类网络，将上一步提取的特征作为网络输入，然后进行训练， 最终在测试集上预测

```
net = nn.Sequential()
with net.name_scope():
    net.add(nn.BatchNorm())
    net.add(nn.Dense(1024))
    net.add(nn.BatchNorm())
    net.add(nn.Activation('relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))

```

## Tips
- 代码使用的是transfer_learning的学习方法， 先在预训练好的模型中提取特征， 多个网络的特征融合之后，再训练分类网络，
这样既能提高准确率快速收敛，还能节省显存，可以在低配的机器中运行
- 网络的选择，代码中选择了inceptionv3和resnet152_v1两个网络，是在大量尝试中得到的，具体请看这个链接[https://github.com/ypwhs/DogBreed_gluon](https://github.com/ypwhs/DogBreed_gluon)
- 训练中的参数设置：
  - 图片输入的size
  - learning的decay策略
- 使用更大的数据集景行数据扩充

