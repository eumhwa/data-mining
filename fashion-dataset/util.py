import os
import random
import numpy as np
import pandas as pd

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from sklearn.metrics import f1_score



def get_train_transform(img_size):
    trans = transforms.Compose(
        [
            #transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return trans

def get_test_transform(img_size):
    trans = transforms.Compose(
        [
            #transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return trans


class FashionDataset(Dataset):
        def __init__(self, df, img_ary, file_path, master_class, sub_class, label_type="g", transform=None):
            self.file_list = [os.path.join(file_path, str(f))+".jpg" for f in df.id]
            self.img_ary = [Image.fromarray(img_ary[ii]) for ii in df.id]
            
            self.label_enc = {"Boys":0, "Girls":1, "Men":2, "Unisex":3, "Women":4}
            self.label_g = [self.label_enc[g] for g in df.gender]
            self.label_m = [master_class.index(m) for m in df.masterCategory]
            self.label_s = [sub_class.index(s) for s in df.subCategory]
            if label_type == "g":
                self.labels = self.label_g
            elif label_type == "m":
                self.labels = self.label_m
            else:
                self.labels = self.label_s

            self.transform = transform
                                                
        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            
            label = int(self.labels[idx])# for zero indexing

            image = (self.img_ary[idx])                                                     
            if self.transform is None:
                self.transform = get_test_transform((224,224))
                image_transform = self.transform(image)
            else:
                image_transform = self.transform(image)
                                                                                                                         
            return image_transform, label


class HCMFashionDataset(Dataset):
    def __init__(self, df, img_ary, file_path, master_class, sub_class, transform=None):
        self.file_list = [os.path.join(file_path, str(f))+".jpg" for f in df.id]
        self.img_ary = [Image.fromarray(img_ary[ii]) for ii in df.id]
        self.label_enc = {"Boys":0, "Girls":1, "Men":2, "Unisex":3, "Women":4}
        #self.label_g = [self.label_enc[g] for g in df.gender]
        self.label_m = [master_class.index(m) for m in df.masterCategory]
        self.label_s = [sub_class.index(s) for s in df.subCategory]
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = self.img_ary[idx] #Image.open(self.file_list[idx]).convert('RGB')
        #label_g = int(self.label_g[idx])
        label_m = int(self.label_m[idx])
        label_s = int(self.label_s[idx])
        
        if self.transform is None:
            self.transform = get_test_transform((224,224))
            image_transform = self.transform(image)
        else:
            image_transform = self.transform(image)
        
        return image_transform, label_m, label_s

        
def performance(ys, ps):
        oneh = []
        for y, p in zip(ys, ps):
            if y==p:
                oneh.append(1)
            else:
                oneh.append(0)
                                                                                
        return 100*(sum(oneh)/len(oneh))


def train_func(model, trainloader, validloader, epch, device='cpu', model_type='vit'):
    print("Training start ...")
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

                                    # epch = self.epoch if self.iteration==1 else self.retrain_epoch
    valid_ys = [l for l in validloader.dataset.labels]
    model.to(device)
    train_accs, valid_accs, train_losses, valid_losses = [], [], [], []
    for e in range(epch):
        train_losses, train_ps, train_ys = [], [], []
        model.train()
                                                                                                        
        for data, y in trainloader:
            data = data.to(device)
            y = y.to(device)
            opt.zero_grad()
            p = model(data).logits if model_type == "vit" else model(data)
            loss = criterion(p, y)
            train_losses.append(loss.cpu().detach().numpy().item())
            train_ps.extend(p.argmax(axis=1).cpu().detach().numpy().tolist())
            train_ys.extend(y.cpu().detach().numpy().tolist())  

            loss.backward()
            opt.step()
        
        valid_losses, valid_ps = [], []
        with torch.no_grad():
            model.eval()
            for data_, y_ in validloader:                                                                                                                                                                                                                                                                                                                                                       
                data_ = data_.to(device)
                y_ = y_.to(device)

                p_ = model(data_).logits if model_type == "vit" else model(data_)
                loss_ = criterion(p_, y_)
                valid_losses.append(loss_.cpu().detach().numpy().item())
                valid_ps.extend(p_.argmax(axis=1).cpu().detach().numpy().tolist())

        if True:
            train_acc = round(performance(train_ps, train_ys), 2)
            valid_acc = round(performance(valid_ps, valid_ys), 2)
            train_loss = round(sum(train_losses)/len(train_losses), 4)
            valid_loss = round(sum(valid_losses)/len(valid_losses), 4)
            print(f"#[Epoch]: {e}/{epch}")
            print(f"## ACC (train, valid): ({str(train_acc)}%, {str(valid_acc)}%)")
            print(f"## Loss (train, valid):({train_loss}, {valid_loss})")
            print("############################################")
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)


    return model, {"train_acc":train_accs, "valid_acc":valid_accs, "train_loss":train_losses, "valid_loss":valid_losses}


class HCM(nn.Module):

    def __init__(self, num_classes=[7,45]):
        super(HCM, self).__init__()

        # encoder = resnet50(pretrained=True, progress=True)
        self.resnet = resnet50(pretrained=True, progress=True) #torch.nn.Sequential(*(list(encoder.children())[:-1]))

        self.linear_master = nn.Linear(1000, num_classes[0])
        self.linear_sub = nn.Linear(1000, num_classes[1])

        self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])


    def forward(self, x):

        x = self.resnet(x)

        level_1 = self.softmax_reg1(self.linear_master(x))
        level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_sub(x)), dim=1))

        return  level_1, level_2

class HierarchicalLossNetwork:

    def __init__(self, master_class, sub_class, hierarchical_labels, device='cpu', total_level=2, alpha=1, beta=0.8, p_loss=3):

        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.level_one_labels, self.level_two_labels = master_class, sub_class
        self.hierarchical_labels = hierarchical_labels
        self.numeric_hierarchy = self.words_to_indices()


    def words_to_indices(self):

        numeric_hierarchy = {}
        for k, v in self.hierarchical_labels.items():
            numeric_hierarchy[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]

        return numeric_hierarchy


    def check_hierarchy(self, current_level, previous_level):

        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [not current_level[i] in self.numeric_hierarchy[previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):


        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
