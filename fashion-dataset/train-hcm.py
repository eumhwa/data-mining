import os, pickle
import numpy as np
import pandas as pd

from torchvision.models import resnet50
from transformers import ViTForImageClassification
from util import *


device = 'cuda:0'
data_path = "/home/ds/eh/mss/fashion-dataset"
file_path = os.path.join(data_path, "images")

epch = 15
batch_size = 32
img_shape = (224, 224)

columns_to_skip = ['productDisplayName']
styles_df = pd.read_csv(os.path.join(data_path, "styles.csv"), usecols=lambda x: x not in columns_to_skip )
styles_df = styles_df.dropna().reset_index(drop=True)


images = os.listdir(os.path.join(data_path, 'images'))
image_id = [int(im.split('.')[0]) for im in images]
styles_df = styles_df.loc[styles_df.id.isin(image_id), ]


master_categories = styles_df.masterCategory.unique().tolist()
sub_categories = styles_df.subCategory.unique().tolist()


random.seed(1000)
ids = styles_df.id.values.copy()
random.shuffle(ids)

with open(file='/home/ds/eh/mss/img_ary.pkl', mode='rb') as f:
    img_ary = pickle.load(f)
    print("img array loaded")

train_idx = ids[:int(styles_df.shape[0]*0.6)]
val_idx =  ids[int(styles_df.shape[0]*0.6):int(styles_df.shape[0]*0.8)]
test_idx = ids[int(styles_df.shape[0]*0.8):]

train = styles_df.loc[styles_df.id.isin(train_idx), ]
val = styles_df.loc[styles_df.id.isin(val_idx), ]
test = styles_df.loc[styles_df.id.isin(test_idx), ]

master_categories = styles_df.masterCategory.unique().tolist()
sub_categories = styles_df.subCategory.unique().tolist()
hirarchy = {}
for m in master_categories:
    sub = styles_df.loc[styles_df.masterCategory == m, ]['subCategory'].unique().tolist()
    hirarchy[m] = sub

train_dset = HCMFashionDataset(train, img_ary, file_path,  master_categories, sub_categories, get_train_transform(img_shape))
valid_dset = HCMFashionDataset(val, img_ary, file_path,  master_categories, sub_categories, get_train_transform(img_shape))
test_dset = HCMFashionDataset(test, img_ary, file_path,  master_categories, sub_categories, get_test_transform(img_shape))

trainloader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
validloader = DataLoader(valid_dset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

model = HCM()

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
HLN = HierarchicalLossNetwork(master_categories, sub_categories, hierarchical_labels=hirarchy, device=device)

train_y_m = [l1 for l1 in trainloader.dataset.label_m]
train_y_s = [l2 for l2 in trainloader.dataset.label_s]

valid_y_m = [l1 for l1 in validloader.dataset.label_m]
valid_y_s = [l2 for l2 in validloader.dataset.label_s]

model.to(device)

print("Training start ...")
for e in range(epch):
    train_losses, train_gender_p,  train_master_p, train_sub_p = [], [], [], []
    model.train()
            
    for data, y_m, y_s in trainloader:
        data = data.to(device)
        y_m = y_m.to(device)
        y_s = y_s.to(device)
        opt.zero_grad()

        p_m, p_s = model(data)
        prediction = [p_m, p_s]
                
        dloss = HLN.calculate_dloss(prediction, [y_m, y_s])
        lloss = HLN.calculate_lloss(prediction, [y_m, y_s])

        total_loss = lloss + dloss 
                
        train_losses.append(total_loss.cpu().detach().numpy().item())
        train_master_p.extend(p_m.argmax(axis=1).cpu().detach().numpy().tolist())
        train_sub_p.extend(p_s.argmax(axis=1).cpu().detach().numpy().tolist())
                
        total_loss.backward()
        opt.step()
                
    valid_losses, valid_gender_p, valid_master_p, valid_sub_p = [], [], [], []
    with torch.no_grad():
        model.eval()
        for data_, ym_, ys_ in validloader:
                    
            data_ = data_.to(device)
            ym_ = ym_.to(device)
            ys_ = ys_.to(device)

            pm_, ps_ = model(data_)
            prediction_ = [pm_, ps_]

            dloss_ = HLN.calculate_dloss(prediction_, [ym_, ys_])
            lloss_ = HLN.calculate_lloss(prediction_, [ym_, ys_])

            total_loss_ = lloss_ + dloss_ 
                    
            valid_losses.append(total_loss_.cpu().detach().numpy().item())
            valid_master_p.extend(pm_.argmax(axis=1).cpu().detach().numpy().tolist())
            valid_sub_p.extend(ps_.argmax(axis=1).cpu().detach().numpy().tolist())
    
    if True:
        train_loss = round(sum(train_losses)/len(train_losses), 4)
        valid_loss = round(sum(valid_losses)/len(valid_losses), 4)

        train_acc_m = round(performance(train_master_p, train_y_m), 2)
        train_acc_s = round(performance(train_sub_p, train_y_s), 2)

        valid_acc_m = round(performance(valid_master_p, valid_y_m), 2)
        valid_acc_s = round(performance(valid_sub_p, valid_y_s), 2)
        print(f"#[Epoch]: {e}/{epch}")
        print(f"#[Loss (train / valid): ({train_loss}, {valid_loss})]")

        print(f"#[ACC-M (train / valid): ({train_acc_m}, {valid_acc_m})]")
        print(f"#[ACC-S (train / valid): ({train_acc_s}, {valid_acc_s})]")
        print("########################################################")

model_save_path = f'/home/ds/eh/mss/hcm_model.pth'
torch.save(model.state_dict(), model_save_path)