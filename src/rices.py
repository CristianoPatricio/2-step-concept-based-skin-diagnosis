import numpy as np
import pickle
import os

class RICES:
    def __init__(self, dataset, split, valid_ids=None) -> None:
        self.dataset = dataset
        self.split = split
        similarity_path = "data/visual_features"

        if self.split is not None:
            with open(os.path.join(similarity_path, f"{self.dataset}_split_{self.split}_ExpLICD_image_features_train.pkl"), "rb") as f:
                self.image_train_feature = pickle.load(f)
            
            with open(os.path.join(similarity_path, f"{self.dataset}_split_{self.split}_ExpLICD_image_features_test.pkl"), "rb") as f:
                self.image_query_feature = pickle.load(f)
        else:
            with open(os.path.join(similarity_path, f"{self.dataset}_ExpLICD_image_features_train.pkl"), "rb") as f:
                self.image_train_feature = pickle.load(f)
            
            with open(os.path.join(similarity_path, f"{self.dataset}_ExpLICD_image_features_test.pkl"), "rb") as f:
                self.image_query_feature = pickle.load(f)
        
        self.valid_ids = valid_ids

    def get_context_keys(self, key, n):
        """Select top n examples with highest similarity with the image feature of query image.

        Args:
            key (int): Query image ID.
            n (int): The number of examples to select.

        Returns:
            list: top n examples
        """
        similarity = np.matmul(np.stack(list(self.image_train_feature.values()), axis=0), np.array(self.image_query_feature[key]))
        similarity_dict = {k: s for (k, s) in zip(self.image_train_feature.keys(), similarity)}
        sorted_similarity_dict = sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True)

        ids = [x for (x, _) in sorted_similarity_dict]

        if len(self.valid_ids) > 0:
            ids = [x for x in ids if x in self.valid_ids]

        return ids[:n]
