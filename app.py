from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
import cv2
import numpy as np
from models.person_counter import PersonCounter, Config
import matplotlib.pyplot as plt
import io
import base64
import os
import logging

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
# counter = PersonCounter(Config())
# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    template_folder=os.path.abspath('templates'),
    static_folder=os.path.abspath('static')
)
# Khởi tạo model với try-catch
try:
    from models.person_counter import PersonCounter, Config
    counter = PersonCounter(Config())
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    counter = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/count', methods=['POST']) 
def count_people():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        
        # Get parameters
        use_sam = request.form.get('use_sam', 'true').lower() == 'true'
        
        # Save uploaded file temporarily
        file = request.files['image']
        temp_path = "temp_image.jpg"
        file.save(temp_path)
        # Read image before processing
        image = cv2.imread(temp_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # def progress_callback(status):
        #     socketio.emit('progress', {'status': status})
          
        
        # Process with selected model
        results = counter.process_image(
            temp_path,  # Truyền đường dẫn thay vì array
            use_sam=use_sam,
            # callback=progress_callback
        )
        
        # Cleanup temp file
        # import os
        # if os.path.exists(temp_path):
        #     os.remove(temp_path)
            
        # Visualization
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
        for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
            x1,y1,x2,y2 = box
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                               fill=False, color='red',
                               linewidth=2, linestyle='--')
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f'{score:.2f}', color='white',
                    bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title(f'Detected {results["count"]} People')
        plt.axis('off')
        plt.tight_layout()
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()
        
        return jsonify({
            'visualization': plot_url,
            'count': results['count'],
            'boxes': results['boxes'].tolist(),
            'scores': results['scores'].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)