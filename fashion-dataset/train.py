import os, pickle
import numpy as np
import pandas as pd

from torchvision.models import resnet50
from transformers import ViTForImageClassification
from util import *

model_type = "resnet50" # "resnet50"
device = 'cuda:0'
label_type = "g" # "m", "s"
n_class_dict = {"g": 5, "m": 7, "s": 45}
n_class = n_class_dict[label_type]
print(f"target label: {label_type} and n_class: {n_class}")

data_path = "/home/ds/eh/mss/fashion-dataset"
file_path = os.path.join(data_path, "images")

batch_size = 32
img_shape = (224, 224)

columns_to_skip = ['productDisplayName']
styles_df = pd.read_csv(os.path.join(data_path, "styles.csv"), usecols=lambda x: x not in columns_to_skip)
styles_df = styles_df.dropna().reset_index(drop=True)


images = os.listdir(file_path)
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


train_dset = FashionDataset(train, img_ary, file_path,  master_categories, sub_categories, label_type, get_train_transform(img_shape))
valid_dset = FashionDataset(val, img_ary, file_path,  master_categories, sub_categories, label_type, get_train_transform(img_shape))
test_dset = FashionDataset(test, img_ary, file_path,  master_categories, sub_categories, label_type, get_test_transform(img_shape))

trainloader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
validloader = DataLoader(valid_dset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)


if model_type == "resnet50":
    model = resnet50(pretrained=True, progress=True)
    model.fc = nn.Linear(2048, n_class)
elif model_type == "vit":
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_attentions=True)
    model.classifier = nn.Linear(768, n_class)

model, res = train_func(model, trainloader, validloader, 15, device, model_type)

model_save_path = f'/home/ds/eh/mss/{label_type}_{model_type}_best.pth'
torch.save(model.state_dict(), model_save_path)


# save
with open(f'/home/ds/eh/mss/{label_type}_{model_type}_result.pkl', 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)