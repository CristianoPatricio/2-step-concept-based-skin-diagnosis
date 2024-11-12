import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

dx_class_label = ['benign_malignant']
mel_class_labels = ['TRBL', 'BDG', 'WLSA', 'ESA', 'GP', 'PV', 'PRL']
nev_class_labels = ['APC', 'MS', 'OPC']
char_class_labels = mel_class_labels + nev_class_labels 
image_size = 224

class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, metadata, img_extension='jpg', index=None, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        self.img_extension = img_extension
        if index is not None:
            self.metadata = self.metadata.loc[index]
        self.y = self.metadata[dx_class_label].values.flatten().astype(int)
        self.transform = transform
        mel_class_labels = ['TRBL', 'BDG', 'WLSA', 'ESA', 'GP', 'PV', 'PRL']
        nev_class_labels = ['APC', 'MS', 'OPC']
        self.char_class_labels = mel_class_labels + nev_class_labels 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # Get sample
        sample = self.metadata.iloc[index]

        # Get image path
        img_path = os.path.join(self.root_dir, sample['image_id'] + "." + self.img_extension)

        #image = Image.open(img_path).convert("RGB")
        y_dx = sample['benign_malignant']
        y_char = torch.tensor([sample[str(derm_con)] for derm_con in self.char_class_labels]).int()
        image_name = sample['image_id']

        if self.transform:
            image = self.transform(image=image)['image']

        return {"img_path": img_path, "class_label": y_dx, "derm_concepts": y_char, "img_id": image_name}
    

if __name__ == "__main__":
    metadata_file = "data/splits/metadata_ham10000_gt.csv"

    metadata = pd.read_csv(metadata_file)

    test_set = metadata[metadata['split'] == 'test']
    train = metadata[metadata['split'] == 'train']
    
    # Drop lesion Ids from train set that are also in test set
    train_set = train[~train['lesion_id'].isin(test_set['lesion_id'])]

    test_set = HAM10000Dataset(root_dir="data/HAM10000/images",
                               metadata=test_set,
                               img_extension='jpg')
    
    dataloader = DataLoader(test_set, shuffle=False, num_workers=4)

    for batch in dataloader:
        breakpoint()
