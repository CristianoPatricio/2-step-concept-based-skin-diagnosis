from torch.utils.data import Dataset

import pandas as pd
import os
import numpy as np

class PH2Dataset(Dataset):
    """
    A custom Pytorch dataset for reading data from a CSV file regarding the PH2 dataset.
    Args:
        csv_file (str): Path to the CSV file containing the data.
        img_extension (str): The file extension of the images.
        path_to_images (str): The path to the images folder.
    Attributes:
        data (DataFrame): Pandas DataFrame containing the data from the CSV file.
    """

    def __init__(self, csv_file, img_extension, path_to_images):        
        self.data = pd.read_csv(csv_file)
        self.img_extension = img_extension
        self.path_to_images = path_to_images

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing information of a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing information about the image[idx]
                img_id (str): The ID of the image.
                img_path (str): The full path of the image. 
                class_label (str): The diagnostic category of the dermoscopic image.
                derm_concepts (list): A list containing binary values indicating the presence/absence of dermoscopic attributes in the image.
        """

        # Get sample
        sample = self.data.iloc[idx]

        # Get image ID
        img_id = sample["images"]

        # Create image_path
        img_path = os.path.join(self.path_to_images, img_id + '.' + self.img_extension)

        # Get class label
        class_label = sample["labels"]

        # Get derm_concepts
        derm_concepts_df = pd.read_csv("data/Derm7pt/splits/dermoscopic_concepts_PH2_Derm7pt.csv")
        derm_concepts_sample = derm_concepts_df.loc[derm_concepts_df.image_id == img_id]
        derm_concepts = np.array([derm_concepts_sample["TPN"],
                         derm_concepts_sample["APN"],
                         derm_concepts_sample["ISTR"],
                         derm_concepts_sample["RSTR"],
                         derm_concepts_sample["RDG"],
                         derm_concepts_sample["IDG"],
                         derm_concepts_sample["BWV"],
                         derm_concepts_sample["RS"]])

        return {"img_id": img_id, "img_path": img_path, "class_label": class_label, "derm_concepts": derm_concepts.squeeze()}