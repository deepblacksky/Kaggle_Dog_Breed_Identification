import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import init

ctx = mx.gpu()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()


def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss / steps, acc / steps

# 载入训练集的特征向量
with h5py.File('features_train_stanford_v2.h5', 'r') as f:
    # with h5py.File('features_train_v2.h5', 'r') as f:
    # features_vgg = np.array(f['vgg'])
    features_resnet152_v1 = np.array(f['resnet'])
    # features_densenet = np.array(f['densenet'])
    features_inceptionv3 = np.array(f['inception'])
    # features_densenet = np.array(f['densenet'])
    features_inceptionv3 = np.array(f['inception'])
    labels = np.array(f['labels'])

features_resnet152_v1 = features_resnet152_v1.reshape(
    features_resnet152_v1.shape[:2])
# print(features_resnet152_v1.shape)
features_inceptionv3 = features_inceptionv3.reshape(
    features_inceptionv3.shape[:2])
# print(features_inceptionv3.shape)
features = np.concatenate(
    [features_resnet152_v1, features_inceptionv3], axis=-1)
# print(features.shape)

X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.001)
dataset_train = gluon.data.ArrayDataset(nd.array(X_train), nd.array(y_train))
dataset_val = gluon.data.ArrayDataset(nd.array(X_val), nd.array(y_val))
batch_size = 128
data_iter_train = gluon.data.DataLoader(
    dataset_train, batch_size, shuffle=True)
data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

# # 定义模型
# net = nn.Sequential()
# with net.name_scope():
#     net.add(nn.Dropout(0.2))
#     net.add(nn.Dense(512, activation='relu'))
#     net.add(nn.Dropout(0.6))
#     net.add(nn.Dense(120))
# net.initialize(init=init.Xavier(), ctx=ctx)

# 模型v2
net = nn.Sequential()
with net.name_scope():
    net.add(nn.BatchNorm())
    net.add(nn.Dense(1024))
    net.add(nn.BatchNorm())
    net.add(nn.Activation('relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))
net.initialize(init=init.Xavier(), ctx=ctx)

# lr = 1e-4
# wd = 1e-5
# lr_decay = 0.1
# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

lr_sch = mx.lr_scheduler.FactorScheduler(step=1500, factor=0.5)
trainer = gluon.Trainer(net.collect_params(), 'adam', {
                        'learning_rate': 1e-3, 'lr_scheduler': lr_sch})

# train
# max_acc = 0.
# epochs = 120
epochs = 300
for epoch in range(epochs):
    # if epoch > 70:
    #     trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4*0.5, 'wd': wd*2})
    # if epoch > 90:
    #     trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4*0.5*0.5, 'wd': wd*2})
    # if epoch == 110:
    #     trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    # if epoch == 180:
    #     trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    val_loss, val_acc = evaluate(net, data_iter_val)
    
    # if val_acc > max_acc:
    #     max_acc = val_acc
    #     net.save_params("net_best.params")
    # else:
    #     net = nn.Sequential()
    #     with net.name_scope():
    #         net.add(nn.Dense(256, activation='relu'))
    #         net.add(nn.Dropout(0.5))
    #         net.add(nn.Dense(120))
    #     net.load_params("net_best.params", ctx=mx.gpu())
    #     lr = lr * 0.9
    #     wd = wd / 0.9
    #     trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    
    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100))

# 载入测试集的特征向量
with h5py.File('features_test_v2.h5', 'r') as f:
    # features_vgg_test = np.array(f['vgg'])
    features_resnet_test = np.array(f['resnet'])
    # features_densenet_test = np.array(f['densenet'])
    features_inception_test = np.array(f['inception'])

features_resnet_test = features_resnet_test.reshape(
    features_resnet_test.shape[:2])
features_inception_test = features_inception_test.reshape(
    features_inception_test.shape[:2])
features_test = np.concatenate(
    [features_resnet_test, features_inception_test], axis=-1)

# 预测
output = nd.softmax(net(nd.array(features_test).as_in_context(ctx))).asnumpy()
df = pd.read_csv(
    '../data/kaggle_dog_breed_identification/sample_submission.csv')
for i, c in enumerate(df.columns[1:]):
    df[c] = output[:, i]
df.to_csv('pred_v10.csv', index=None)

# pred.csv: 最开始网络结构，只有一个dropout
# pred_v2: 新的网络结构，两个dropout
# pred_v3: 新的结构，图片的size是288和363
# pred_v4: 新结构，正常size，Stanford数据集
# pred_v5: 新结构，size是288和363，Stanford数据集
# pred_v6: 新结构，size是288和363，Stanford数据集，新learning_rate:110 and 150，epochs=300
# pred_v7: 新结构，size是288和363，Stanford数据集，新learning_rate:lr_scheduler，epochs=100
# pred_v8: 模型v2，size是288和363，Stanford数据集，新learning_rate:lr_scheduler，epochs=100
# pred_v9: 模型v2，size是288和363，Stanford数据集，新learning_rate:110 and 180，epochs=300
# pred_v10: 模型v2，size是288和363，Stanford数据集，新learning_rate:lr_scheduler，epochs=300

