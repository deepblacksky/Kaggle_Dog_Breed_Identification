import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

ctx = mx.gpu()

preprocessing = [
    image.ForceResizeAug((224,224)),
    image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]), std=nd.array([0.229, 0.224, 0.225]))
]

def transform(data, label):
    data = data.astype('float32') / 255
    for pre in preprocessing:
        data = pre(data)
    
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

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

# preprocessing[0] = image.ForceResizeAug((224, 224))
preprocessing[0] = image.ForceResizeAug((288, 288))
imgs = vision.ImageFolderDataset('for_train_stanford', transform=transform)
# imgs = vision.ImageFolderDataset('for_train', transform=transform)
data = gluon.data.DataLoader(imgs, 16)
# features_vgg, labels = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet152_v1, labels = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
# features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)

# preprocessing[0] = image.ForceResizeAug((299, 299))
preprocessing[0] = image.ForceResizeAug((363, 363))
imgs_299 = vision.ImageFolderDataset('for_train_stanford', transform=transform)
# imgs_299 = vision.ImageFolderDataset('for_train', transform=transform)
data_299 = gluon.data.DataLoader(imgs_299, 16)
features_inceptionv3, _ = get_features(models.inception_v3(pretrained=True, ctx=ctx), data_299)

with h5py.File('features_train_stanford_v2.h5', 'w') as f:
    # f['vgg'] = features_vgg
    f['resnet'] = features_resnet152_v1
    # f['densenet'] = features_densenet
    f['inception'] = features_inceptionv3
    f['labels'] = labels


# preprocessing[0] = image.ForceResizeAug((224,224))
preprocessing[0] = image.ForceResizeAug((288, 288))
imgs = vision.ImageFolderDataset('for_test', transform=transform)
data = gluon.data.DataLoader(imgs, 16)
# features_vgg, _ = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet152_v1, _ = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
# features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)


# preprocessing[0] = image.ForceResizeAug((299,299))
preprocessing[0] = image.ForceResizeAug((363, 363))
imgs_299 = vision.ImageFolderDataset('for_test', transform=transform)
data_299 = gluon.data.DataLoader(imgs_299, 16)
features_inceptionv3, _ = get_features(models.inception_v3(pretrained=True, ctx=ctx), data_299)

with h5py.File('features_test_v2.h5', 'w') as f:
    # f['vgg'] = features_vgg
    f['resnet'] = features_resnet152_v1
    # f['densenet'] = features_densenet
    f['inception'] = features_inceptionv3


