import sys
import os
from optparse import OptionParser
import copy

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image

from torchvision import transforms
from scipy.special import softmax

# Get the path to dir_a
dir_explicd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'Explicd'))

# Add dir_a to sys.path
sys.path.append(dir_explicd_path)

# Now you can import the class
from model import ExpLICD

explicid_isic_dict = {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']
}

class Explicd:
    """
    Paper: https://arxiv.org/abs/2406.05596
    Code: https://github.com/yhygao/Explicd
    """

    def __init__(self, config) -> None:
        
        self.model = ExpLICD(concept_list=explicid_isic_dict, model_name='biomedclip', config=config)
        
        # We find using orig_in21k vit weights works better than biomedclip vit weights
        # Delete the following if want to use biomedclip weights
        vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True, num_classes=config.num_class)
        vit.head = nn.Identity()
        self.model.model.visual.trunk.load_state_dict(vit.state_dict())

        if config.load:
            self.model.load_state_dict(torch.load(config.load, weights_only=True))
            print('Model loaded from {}'.format(config.load))

        self.model.cuda()

    def get_concept_predictions(self, batch, config):
        template = """The color is {}, the shape is {}, the border is {}, the dermoscopic patterns are {}, the texture is {}, the symmetry is {}, the elevation is {}."""

        val_transforms = copy.deepcopy(config.preprocess)
        val_transforms.transforms.insert(0, transforms.ToPILImage())

        imgs = val_transforms(np.asarray([Image.open(x) for x in batch["img_path"]]).squeeze())
        batch["data"] = imgs.unsqueeze(dim=0)
    
        dict_data = dict()
        with torch.no_grad():    
            img_id = batch["img_id"][0]
            data, label = batch["data"].cuda(), torch.tensor(batch["class_label"]).long().cuda()
            _, image_logits_dict = self.model(data)

            # Get concept predictions
            concept_preds = []
            raw_scores = []
            for key in self.model.concept_token_dict.keys():
                scores = softmax(image_logits_dict[key].cpu().numpy())
                raw_scores.extend(scores.flatten())
                # Get corresponding description
                description = self.model.concept_list[key][scores.argmax()]
                concept_preds.append(description)

            dict_data[img_id] = template.format(*concept_preds)
        
        return dict_data[img_id], raw_scores
    
    def get_label_predictions(self, batch, config):
        labels = ["MEL", "NEV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

        val_transforms = copy.deepcopy(config.preprocess)
        val_transforms.transforms.insert(0, transforms.ToPILImage())

        imgs = val_transforms(np.asarray([Image.open(x) for x in batch["img_path"]]).squeeze())
        batch["data"] = imgs.unsqueeze(dim=0)
    
        with torch.no_grad():    
            img_id = batch["img_id"][0]
            data, _ = batch["data"].cuda(), torch.tensor(batch["class_label"]).long().cuda()
            cls_logits, _ = self.model(data)
            logits = cls_logits.detach().cpu().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        
        if labels is not None:
            return labels[sorted_indices[0][0]], logits[0][sorted_indices[0][0]]
        else:
            return logits

    def get_image_features(self, batch, config):
        val_transforms = copy.deepcopy(config.preprocess)
        val_transforms.transforms.insert(0, transforms.ToPILImage())

        imgs = val_transforms(np.asarray([Image.open(x) for x in batch["img_path"]]).squeeze())
        batch["data"] = imgs.unsqueeze(dim=0)
    
        dict_data = dict()
        with torch.no_grad():    
            img_id = batch["img_id"][0]
            data, label = batch["data"].cuda(), torch.tensor(batch["class_label"]).long().cuda()

            visual_features = self.model.model.visual(data)

        return visual_features.detach().cpu().numpy()
    
    def generate_heatmap(self, batch, config, head="mean"):
        val_transforms = copy.deepcopy(config.preprocess)
        val_transforms.transforms.insert(0, transforms.ToPILImage())

        imgs = val_transforms(np.asarray([Image.open(x) for x in batch["img_path"]]).squeeze())
        batch["data"] = imgs.unsqueeze(dim=0)

        with torch.no_grad():    
            img_id = batch["img_id"][0]
            data, label = batch["data"].cuda(), torch.tensor(batch["class_label"]).long().cuda()
            _, _, img_feat_map, visual_tokens, feat_map = self.model(data)
        
        original_image = cv2.imread(batch["img_path"][0])
        resized_image = cv2.resize(original_image, (224, 224))

        # Average visual concept tokens
        #visual_tokens = visual_tokens.mean(axis=1) # (1,768)
        visual_tokens = visual_tokens[:,1,:]
        
        # Weighted image feature map with visual concept tokens
        #avg_visual_feat = visual_tokens * img_feat_map.squeeze() # (196, 768)
        avg_visual_feat = feat_map.squeeze() # (196, 768)


        # Step 1: Reshape Feature Map to Spatial Grid
        if head == "mean":
            heatmap = avg_visual_feat.mean(axis=1).view(14, 14).detach().cpu().numpy()  # Mean over all heads
        elif head == "middle":
            heatmap = avg_visual_feat[:,int(avg_visual_feat.shape[1]/2)-1].view(14, 14).detach().cpu().numpy()  # Middle head
        elif head == "last":
            heatmap = avg_visual_feat[:,-1].view(14, 14).detach().cpu().numpy()  # Last heads

        # Step 2: Normalize Heatmap for Visualization
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

        # Step 3: Resize Heatmap to Match Image Dimensions
        heatmap_resized = cv2.resize(heatmap, (resized_image.shape[1], resized_image.shape[0]))
        #heatmap_resized[heatmap_resized < 0.5] = 0

        # Step 4: Convert Heatmap to Colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Step 5: Overlay Heatmap on Original Image
        overlay_image = cv2.addWeighted(resized_image, 0.6, heatmap_colored, 0.2, 0)

        # Step 6: Visualize the Result
        """ plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("Heatmap")
        plt.imshow(heatmap_resized, cmap='jet')
        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.savefig("heatmap.png") """

        return resized_image, heatmap_resized, overlay_image

if __name__ == "__main__":
    """
        DEBUG
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
            type='int', help='batch size')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='adamw', type='str')
    parser.add_option('-l', '--lr', dest='lr', default=0.0001, 
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            #default='/data/yunhe/Liver/auto-aug/checkpoint/', help='checkpoint path')
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', 
            default='./log/', help='log path')
    parser.add_option('-m', '--model', type='str', dest='model',
            default='explicd', help='use which model')
    parser.add_option('--linear-probe', dest='linear_probe', action='store_true', help='if use linear probe finetuning')
    parser.add_option('-d', '--dataset', type='str', dest='dataset', 
            default='isic2018', help='name of dataset')
    parser.add_option('--data-path', type='str', dest='data_path', 
            default='/data/local/yg397/dataset/isic2018/', help='the path of the dataset')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='test', help='name prefix')
     

    parser.add_option('--flag', type='int', dest='flag', default=2)

    parser.add_option('--gpu', type='str', dest='gpu',
            default='0')
    parser.add_option('--amp', action='store_true', help='if use mixed precision training')

    (config, args) = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/'
    
    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
    }

    cls_weight_dict = {
        'isic2018': [1, 0.5, 1.2, 1.3, 1, 2, 2], 
    }
    
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]
    
    model = Explicd(config=config)

    val_transforms = copy.deepcopy(config.preprocess)
    val_transforms.transforms.insert(0, transforms.ToPILImage())

    batch = {
        "data": val_transforms(np.asarray(Image.open("data/Derm7pt/images/Nfl040.jpg"))).unsqueeze(dim=0),
        "img_id": "Nfl040",
        "class_label": torch.tensor([0])
    }

    template = model.get_concept_predictions(batch=batch)