import os, pickle
import numpy as np
import pandas as pd

from torchvision.models import resnet50
from transformers import ViTForImageClassification
from util import *

device = 'cuda:0'
model_save_path = "/home/ds/eh/mss/hcm_model.pth"
data_path = "/home/ds/eh/mss/fashion-dataset"

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

print(len(train_idx), len(val_idx), len(test_idx))

train = styles_df.loc[styles_df.id.isin(train_idx), ]
val = styles_df.loc[styles_df.id.isin(val_idx), ]
test = styles_df.loc[styles_df.id.isin(test_idx), ]

print(train.shape, val.shape, test.shape)

file_path = os.path.join(data_path, "images")


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
model.load_state_dict(torch.load(model_save_path))
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
HLN = HierarchicalLossNetwork(master_categories, sub_categories, hierarchical_labels=hirarchy, device=device)


test_y_m = [l1 for l1 in test_loader.dataset.label_m]
test_y_s = [l2 for l2 in test_loader.dataset.label_s]

test_losses,  test_master_p, test_sub_p, test_master_f1, test_sub_f1 = [], [], [], [], []
with torch.no_grad():
    model.eval()
    for data_, ym_, ys_ in test_loader:
                    
        data_ = data_.to(device)
        ym_ = ym_.to(device)
        ys_ = ys_.to(device)

        pm_, ps_ = model(data_)
        prediction_ = [pm_, ps_]

        dloss_ = HLN.calculate_dloss(prediction_, [ym_, ys_])
        lloss_ = HLN.calculate_lloss(prediction_, [ym_, ys_])

        total_loss_ = lloss_ + dloss_ 
                    
        test_losses.append(total_loss_.cpu().detach().numpy().item())
        test_master_p.extend(pm_.argmax(axis=1).cpu().detach().numpy().tolist())
        test_sub_p.extend(ps_.argmax(axis=1).cpu().detach().numpy().tolist())
        f1_m = f1_score(pm_.argmax(axis=1).cpu().detach().numpy().tolist(), ym_.cpu(), average='macro')
        f1_s = f1_score(ps_.argmax(axis=1).cpu().detach().numpy().tolist(), ys_.cpu(), average='macro')
        test_master_f1.append(f1_m)
        test_sub_f1.append(f1_s)

    
if True:
    test_loss = round(sum(test_losses)/len(test_losses), 4)

    test_acc_m = round(performance(test_master_p, test_y_m), 2)
    test_acc_s = round(performance(test_sub_p, test_y_s), 2)
    test_f1_m = round(sum(test_master_f1)/len(test_master_f1), 4)
    test_f1_s = round(sum(test_sub_f1)/len(test_sub_f1), 4)

res = {"test_acc_m":test_acc_m, "test_acc_s":test_acc_s, "test_f1_m":test_f1_m, "test_f1_s":test_f1_s, "test_loss":test_loss}
# save
with open(f'/home/ds/eh/mss/hcm_result.pkl', 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
