# agi_core/perception/vision.py
"""
Vision perception module
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List, Any
import numpy as np
import cv2

from utils.logger import logger

class VisionPerception:
    """Vision perception module with object detection and scene understanding"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize models
        self._init_models()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _init_models(self):
        """Initialize vision models"""
        
        # Object detection (YOLO placeholder)
        try:
            # Try to import torchvision detection models
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            
            self.detection_model = fasterrcnn_resnet50_fpn(
                pretrained=True,
                min_size=224,
                max_size=224
            ).to(self.device).eval()
            
            logger.info("Loaded object detection model")
            
        except Exception as e:
            logger.warning(f"Could not load detection model: {e}")
            self.detection_model = None
            
        # Scene classification
        try:
            from torchvision.models import resnet50
            
            self.classification_model = resnet50(pretrained=True).to(self.device).eval()
            logger.info("Loaded scene classification model")
            
        except Exception as e:
            logger.warning(f"Could not load classification model: {e}")
            self.classification_model = None
            
        # Optical flow
        self.flow_model = None
        
    def perceive(self, image: Image.Image) -> Dict[str, Any]:
        """Perceive image and extract features"""
        
        logger.debug("Processing image perception")
        
        perception = {
            'objects': [],
            'scene': {},
            'features': None,
            'optical_flow': None
        }
        
        # Object detection
        if self.detection_model is not None:
            objects = self.detect_objects(image)
            perception['objects'] = objects
            
        # Scene classification
        if self.classification_model is not None:
            scene = self.classify_scene(image)
            perception['scene'] = scene
            
        # Extract features
        features = self.extract_features(image)
        perception['features'] = features
        
        return perception
        
    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        
        if self.detection_model is None:
            return []
            
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.detection_model(image_tensor)[0]
            
        objects = []
        
        for i in range(len(predictions['boxes'])):
            if predictions['scores'][i] > 0.5:  # Confidence threshold
                obj = {
                    'bbox': predictions['boxes'][i].cpu().numpy().tolist(),
                    'label': predictions['labels'][i].item(),
                    'score': predictions['scores'][i].item(),
                    'category': self._get_category(predictions['labels'][i].item())
                }
                objects.append(obj)
                
        logger.debug(f"Detected {len(objects)} objects")
        
        return objects
        
    def classify_scene(self, image: Image.Image) -> Dict[str, Any]:
        """Classify scene"""
        
        if self.classification_model is None:
            return {}
            
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.classification_model(image_tensor)
            
        # Get top predictions
        probabilities = torch.softmax(features, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Load ImageNet labels (simplified)
        labels = self._get_imagenet_labels()
        
        scene = {
            'predictions': []
        }
        
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = labels.get(idx.item(), f"class_{idx.item()}")
            scene['predictions'].append({
                'label': label,
                'confidence': prob.item()
            })
            
        # Main scene label
        if scene['predictions']:
            scene['main'] = scene['predictions'][0]['label']
            
        return scene
        
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract visual features"""
        
        if self.classification_model is None:
            # Return random features as placeholder
            return np.random.randn(1, 2048)
            
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features from penultimate layer
        with torch.no_grad():
            # Forward pass until penultimate layer
            x = self.classification_model.conv1(image_tensor)
            x = self.classification_model.bn1(x)
            x = self.classification_model.relu(x)
            x = self.classification_model.maxpool(x)
            
            x = self.classification_model.layer1(x)
            x = self.classification_model.layer2(x)
            x = self.classification_model.layer3(x)
            x = self.classification_model.layer4(x)
            
            x = self.classification_model.avgpool(x)
            features = torch.flatten(x, 1)
            
        return features.cpu().numpy()
        
    def track_objects(self, prev_frame: np.ndarray, 
                     curr_frame: np.ndarray) -> Dict[str, Any]:
        """Track objects between frames"""
        
        # Simple optical flow using OpenCV
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return {
            'flow': flow,
            'magnitude_mean': np.mean(magnitude),
            'angle_mean': np.mean(angle),
            'movement': magnitude > 0.1  # Binary movement mask
        }
        
    def _get_category(self, label_id: int) -> str:
        """Convert label ID to category name"""
        
        categories = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
            5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
            9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            # ... more categories
        }
        
        return categories.get(label_id, f'object_{label_id}')
        
    def _get_imagenet_labels(self) -> Dict[int, str]:
        """Get ImageNet labels"""
        
        # Simplified - in practice would load from file
        return {
            0: 'tench', 1: 'goldfish', 2: 'great white shark',
            # ... more labels
        }

class VisualMemory:
    """Visual memory for storing and retrieving visual information"""
    
    def __init__(self, config):
        self.config = config
        self.visual_features = []
        self.visual_metadata = []
        
    def store(self, image_features: np.ndarray, metadata: Dict[str, Any]):
        """Store visual features with metadata"""
        
        self.visual_features.append(image_features)
        self.visual_metadata.append(metadata)
        
    def retrieve_similar(self, query_features: np.ndarray, 
                        k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar visual memories"""
        
        if not self.visual_features:
            return []
            
        # Calculate similarities
        similarities = []
        for features in self.visual_features:
            sim = np.dot(query_features.flatten(), features.flatten())
            sim /= (np.linalg.norm(query_features) * np.linalg.norm(features) + 1e-8)
            similarities.append(sim)
            
        # Get top-k similar
        indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in indices:
            results.append({
                'metadata': self.visual_metadata[idx],
                'similarity': similarities[idx]
            })
            
        return results