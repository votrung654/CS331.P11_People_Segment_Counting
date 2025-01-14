import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

class Config:
    def __init__(self):
        self.model_dir = "D:/Downloads/CS331_project/demo_app/models"
        self.sam_path = os.path.join(self.model_dir, "sam_vit_h_4b8939.pth")
        self.score_threshold = 0.7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_points = 100
        self.coarse_thresh = 0.7
        self.fine_thresh = 0.3

class PersonCounter:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.device
        
        # Initialize SAM
        self.sam = sam_model_registry["vit_h"](checkpoint=self.config.sam_path)
        self.sam_predictor = SamPredictor(self.sam)
        self.sam.to(self.device)

        # Initialize Mask R-CNN 
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.mask_rcnn = maskrcnn_resnet50_fpn(
            weights=weights,
            box_detections_per_img=100,
            rpn_pre_nms_top_n_test=6000,
            rpn_post_nms_top_n_test=1000,
        )
        in_features = self.mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        self.mask_rcnn.to(self.device)
        self.mask_rcnn.eval()

    def process_image(self, image_path, use_sam=True, callback=None):
        """Process image with progress tracking"""
        if callback: callback("Loading image...")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        h, w = image.shape[:2]
        if max(h,w) > 1024:
            scale = 1024/max(h,w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
            
        image_tensor = torch.from_numpy(image).permute(2,0,1).float()[None] / 255.0
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if use_sam:
                return self.process_with_sam(image, image_tensor, callback)
            else:
                return self.process_maskrcnn_only(image, image_tensor, callback)

    def process_with_sam(self, image, image_tensor, callback):
        """Combined SAM + Mask R-CNN approach"""
        if callback: callback("Detecting people...")
        predictions = self.mask_rcnn(image_tensor)[0]
        
        labels = predictions['labels']  # Thêm dòng này
        scores = predictions['scores']

        # Filter chỉ lấy person (label = 0 trong COCO)
        person_mask = labels == 0  # Thêm dòng này``
        high_score_mask = scores > self.config.score_threshold
        # Combine masks
        final_mask = person_mask & high_score_mask  # Thêm dòng này
        # final_mask = person_mask
        boxes = predictions['boxes'][final_mask]
        filtered_scores = scores[final_mask]
            
        return {
            'count': len(boxes),
            'boxes': boxes.cpu().numpy(), 
            'scores': filtered_scores.cpu().numpy(),
            'image': image
        }

    def process_maskrcnn_only(self, image, image_tensor, callback):
        """Mask R-CNN only approach"""
        if callback: callback("Detecting people...")
        predictions = self.mask_rcnn(image_tensor)[0]
        
        labels = predictions['labels']  # Thêm dòng này
        scores = predictions['scores']

        # Filter chỉ lấy person (label = 1 trong COCO)
        person_mask = labels == 1  # Thêm dòng này``
        high_score_mask = scores > self.config.score_threshold
        # Combine masks
        # final_mask = person_mask & high_score_mask  # Thêm dòng này
        final_mask = person_mask
        boxes = predictions['boxes'][final_mask]
        filtered_scores = scores[final_mask]
            
        return {
            'count': len(boxes),
            'boxes': boxes.cpu().numpy(), 
            'scores': filtered_scores.cpu().numpy(),
            'image': image
        }

    def visualize_results(self, results, save_path=None):
        """Visualize detection results"""
        plt.figure(figsize=(12,8))
        plt.imshow(results['image'])
        
        # Draw boxes
        for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f'{score:.2f}',
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5))
            
        # Draw points if available
        if 'points' in results:
            points = results['points']
            plt.scatter(points[:,0], points[:,1], 
                       c='yellow', s=20, marker='*')
            
        plt.title(f'Detected People: {results["count"]}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()