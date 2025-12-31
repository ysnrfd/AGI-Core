# agi_core/utils/embeddings.py
"""
Embedding utilities for memory and perception
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional, Any
import numpy as np

class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32) -> np.ndarray:
        """Encode text(s) to embeddings"""
        
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.config.hidden_size

class MultiModalEmbeddings:
    """Handle multimodal embeddings (text, vision, audio)"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize text encoder
        self.text_encoder = EmbeddingModel(device=self.device)
        
        # Vision encoder (CLIP-like)
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model.eval()
        except:
            self.vision_model = None
            self.vision_processor = None
            
        # Audio encoder placeholder
        self.audio_encoder = None
        
    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        """Encode image to embedding"""
        if self.vision_model is None:
            raise ValueError("Vision model not available")
            
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.vision_model.get_image_features(**inputs)
        return outputs.cpu().numpy()
        
    def encode_audio(self, audio) -> np.ndarray:
        """Encode audio to embedding"""
        # Placeholder - would implement with Wav2Vec2 or similar
        if self.audio_encoder is None:
            return np.random.randn(1, 768)  # Placeholder
            
        # TODO: Implement actual audio encoding
        pass
        
    def encode_multimodal(self, text: Optional[str] = None,
                         image: Optional[Any] = None,
                         audio: Optional[Any] = None) -> np.ndarray:
        """Encode multimodal input"""
        embeddings = []
        
        if text is not None:
            text_emb = self.text_encoder.encode(text)
            embeddings.append(text_emb)
            
        if image is not None and self.vision_model is not None:
            image_emb = self.encode_image(image)
            embeddings.append(image_emb)
            
        if audio is not None and self.audio_encoder is not None:
            audio_emb = self.encode_audio(audio)
            embeddings.append(audio_emb)
            
        if embeddings:
            # Concatenate or average based on modality
            return np.concatenate(embeddings, axis=1)
        else:
            raise ValueError("No input provided")