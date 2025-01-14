class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model paths
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        
        # Image processing
        self.image_size = 640
        self.confidence_threshold = 0.5
        
        # Point detection
        self.num_points = 100
        
        # Mask thresholds
        self.coarse_threshold = 0.7
        self.fine_threshold = 0.3