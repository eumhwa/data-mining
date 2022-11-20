import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import torchvision.transforms as transforms

data_path = "/home/ds/eh/mss/fashion-dataset"
columns_to_skip = ['productDisplayName']
styles_df = pd.read_csv(os.path.join(data_path, "styles.csv"), usecols=lambda x: x not in columns_to_skip )

images = os.listdir(os.path.join(data_path, 'images'))
image_id = [int(im.split('.')[0]) for im in images]
styles_df = styles_df.loc[styles_df.id.isin(image_id), ]


img_dict = {}
for i, id in enumerate(styles_df.id):
    image_path = data_path + "/images/" + str(id) + ".jpg"
    img = Image.open(image_path).convert('RGB')
    img = transforms.Resize((224,224))(img)
    img_ary = np.asarray(img)
    img_dict[id] = img_ary

    if i % 500 == 0:
        print(f"[{i}/{styles_df.shape[0]}] processed")

with open('/home/ds/eh/mss/img_ary.pkl', 'wb') as f:
    pickle.dump(img_dict, f, pickle.HIGHEST_PROTOCOL)