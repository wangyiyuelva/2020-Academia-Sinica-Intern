# 基礎資料處理切分
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Deep Learning 相關
import torch
from torch import nn, optim, utils

# 系統互動
import time
import argparse

# 自訂部分
from model import ResNet50MothClassifier
from dataset import ImageDatasetFromFileSpecial
from average_meter import AverageMeter

# 從命令列讀參數
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="/Users/gsmai/Documents/GitHub/vae/save/downloaded256", type=str, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
opt = parser.parse_args()

# 讀取資料 & 前處理
dataset_path = 'sp_all.csv'
df = pd.read_csv(dataset_path, sep="\t")

# 移除未鑑定到種的資料
df = df[~df.Species.isna()].reset_index(drop=True)

# 造出物種清單，並產生每筆資料對應的物種 id, 視為 classification 用的 target y
species_list, species_id = np.unique(df.Species, return_inverse=True)
y = species_id

print([len(species_list), len(y)])

# 產生影像路徑
x = img_paths = (opt.dataroot + '/' + df.Number + '.jpg').values


# 切 train, valid, test
x_train_valid, x_test, y_train_valid, y_test = train_test_split(x, y,  train_size=.8, test_size=.2, random_state=5566)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid,  train_size=.8, test_size=.2, random_state=5566)

train_set = ImageDatasetFromFileSpecial(x_train, '', y=y_train, aug=True)
train_data_loader = utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True)

# valid 與 test 時不需要做 augmentation
valid_set = ImageDatasetFromFileSpecial(x_valid, '', y=y_valid, aug=False)
valid_data_loader = utils.data.DataLoader(valid_set, batch_size=opt.batchSize, shuffle=False)

test_set = ImageDatasetFromFileSpecial(x_test, '', y=y_test, aug=False, return_filename=True)
test_data_loader = utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False)

valid_and_test_set = ImageDatasetFromFileSpecial(np.concatenate([x_valid, x_test]), '', y=np.concatenate([y_valid, y_test]), aug=False, return_filename=True)
valid_and_test_data_loader = utils.data.DataLoader(valid_and_test_set, batch_size=opt.batchSize, shuffle=False)

def load_model(model, pretrained):
    weights = torch.load(pretrained, map_location='cuda:0')
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

pretrained = "./resnet50_moth_classifier.pth"

# init model, 把模型搬進 GPU:0 的記憶體中
model = ResNet50MothClassifier(num_of_classes=len(species_list)).to('cuda:0')
load_model(model, pretrained)
model = model.eval()

# 設定分類器用的 cross entropy loss
cross_entropy = nn.CrossEntropyLoss()

preds_id = []
preds_name = []
fnames = []
labels_id = []
labels_name = []
probs = []

test_loss = AverageMeter()

start_time = time.time()
for iteration, (imgs, label, fname_) in enumerate(valid_and_test_data_loader, 0):

    print(iteration, end='\r')
    imgs_cuda = imgs.cuda()
    pred_ = model(imgs_cuda)
    label_cuda = label.to('cuda:0')
    loss = cross_entropy(pred_, label_cuda)

    pred_softmax_numpy = nn.functional.softmax(pred_).data.clone().detach().cpu().numpy()
    pred = np.argmax(pred_softmax_numpy, axis=1)
    pred_prob = np.round(np.take_along_axis(pred_softmax_numpy, pred.reshape(-1,1), axis=1).reshape(-1) * 100 , 2)

    pred_name = species_list[pred]
    label_name = species_list[np.array(label)]

    fname = [f.split('/')[-1] for f in fname_]

    fnames.append(fname)
    preds_id.append(pred)
    labels_id.append(label)
    preds_name.append(pred_name)
    labels_name.append(label_name)
    probs.append(pred_prob)
    
    test_loss.update(loss.item())

    del imgs_cuda, pred_, label_cuda, loss

print()
print ("%.2f" % (time.time() - start_time))
print(test_loss.avg)

fnames_unions = np.concatenate(fnames)
preds_id_unions = np.concatenate(preds_id)
labels_id_unions = np.concatenate(labels_id)
preds_name_unions = np.concatenate(preds_name)
labels_name_unions = np.concatenate(labels_name)
probs_unions = np.concatenate(probs)

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
accuracy_score(labels_id_unions, preds_id_unions)
balanced_accuracy_score(labels_id_unions, preds_id_unions)
f1_score(labels_id_unions, preds_id_unions, average='weighted')

results = pd.DataFrame({'file': fnames_unions, 'true_id': labels_id_unions, 'pred_id': preds_id_unions, 'true_name': labels_name_unions, 'pred_name': preds_name_unions, 'probs': probs_unions})
results.to_csv('valid_and_test_moth_classification.csv', sep='\t', index=False)