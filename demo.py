import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from models.person_counter import PersonCounter, Config
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class PersonCounterApp:
    def __init__(self):
        plt.ion()
        try:
            self.counter = PersonCounter(Config())
            self.setup_gui()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise e
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Person Counter Demo")
        self.root.geometry("400x300")
        
        # Model selection
        model_frame = ttk.LabelFrame(self.root, text="Select Model", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="sam")
        ttk.Radiobutton(model_frame, text="SAM + Mask R-CNN", 
                       value="sam", variable=self.model_var).pack(side="left", padx=5)
        ttk.Radiobutton(model_frame, text="Mask R-CNN only", 
                       value="maskrcnn", variable=self.model_var).pack(side="left", padx=5)
        
        # Upload button
        ttk.Button(self.root, text="Upload Image", 
                  command=self.process_image).pack(pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
        
    def process_image(self):
        # Get image file
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        self.status_var.set("Processing...")
        self.root.update()
        
        # Process image
        use_sam = self.model_var.get() == "sam"
        results = self.counter.process_image(file_path, use_sam=use_sam)
        
        # Display results
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15,5))
        
        # Original Image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        if use_sam:
            # Points & Masks
            plt.subplot(132)
            plt.imshow(image)
            plt.imshow(results['masks'].any(0), alpha=0.3, cmap='jet')
            plt.scatter(results['points'][:,0], results['points'][:,1],
                      c='red', s=30, marker='*')
            plt.title('Points & Masks')
            plt.axis('off')
            
        # Final Detections
        plt.subplot(133 if use_sam else 132)
        plt.imshow(image)
        for box, score in zip(results['boxes'], results['scores']):
            x1,y1,x2,y2 = box
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f'{score:.2f}', color='white',
                    bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title(f'Detected {results["count"]} People')
        plt.axis('off')
        plt.tight_layout()
        
        self.status_var.set(f"Detected {results['count']} people")
        plt.draw()
        plt.pause(0.001)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PersonCounterApp()
    app.run()