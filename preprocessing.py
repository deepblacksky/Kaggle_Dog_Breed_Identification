import pandas as pd
import shutil
import os

data_dir = '/home/yuxin/DL/MXNET_GLUON/data/kaggle_dog_breed_identification'

# 在for_train文件夹对应的类别下存放训练图片的软链接
df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
path = 'for_train'

if os.path.exists(path):
    shutil.rmtree(path)

for i, (fname, breed) in df.iterrows():
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    # e.g. for_train/boston_bull/000bec180eb18c7604dcecc8fe0dba07.jpg
    os.symlink(os.path.join(data_dir, 'train/%s.jpg' % fname), '%s/%s.jpg' % (path2, fname))

# 在for_test文件夹下存放测试图片的软链接
df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
path = 'for_test'
breed = '0'

if os.path.exists(path):
    shutil.rmtree(path)

for fname in df['id']:
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    ## e.g. for_test/0/00a3edd22dc7859c487a64777fc8d093.jpg
    os.symlink(os.path.join(data_dir, 'test/%s.jpg' % fname), '%s/%s.jpg' % (path2, fname))

# 在for_train_stanford文件夹对应的类别下存放训练图片和standford dataset的软链接
df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
path = 'for_train_stanford'

if os.path.exists(path):
    shutil.rmtree(path)

for i, (fname, breed) in df.iterrows():
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    # e.g. for_train_stanford/boston_bull/000bec180eb18c7604dcecc8fe0dba07.jpg
    os.symlink(os.path.join(data_dir, 'train/%s.jpg' % fname), '%s/%s.jpg' % (path2, fname))     
# stanford dog dataset
list_breed = os.listdir(os.path.join(data_dir, 'Images'))
for breed in list_breed:
    list_img = os.listdir(os.path.join(data_dir, 'Images/%s' % breed))
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    for img in list_img:
        os.symlink(os.path.join(data_dir, 'Images/%s/%s' % (breed, img)), '%s/%s' % (path2, img))

