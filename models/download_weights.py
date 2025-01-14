import torch
import wget
import os

def download_models():
    """Download pre-trained models"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Mask R-CNN from torchvision
    maskrcnn = torch.hub.load('pytorch/vision:v0.10.0', 
                             'maskrcnn_resnet50_fpn', 
                             pretrained=True)
    torch.save(maskrcnn.state_dict(), 
              os.path.join(models_dir, "maskrcnn_coco.pth"))
    
    # SAM model
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_path = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_path):
        wget.download(sam_url, sam_path)

if __name__ == "__main__":
    download_models()