"""
Utility functions for image preprocessing and prediction formatting.
"""
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)


def get_top_predictions(
    probabilities: torch.Tensor,
    classes: List[str]
) -> Tuple[Dict[str, any], List[Dict[str, any]]]:
    """
    Get top-1 and top-5 predictions from model output.
    
    Args:
        probabilities: Tensor of shape (1, num_classes) with softmax probabilities
        classes: List of class names
        
    Returns:
        Tuple of (top1_dict, top5_list)
    """
    # Get top 5 indices
    top5_probs, top5_indices = torch.topk(probabilities[0], k=min(5, len(classes)))
    
    # Top-1
    top1_idx = top5_indices[0].item()
    top1 = {
        "label": classes[top1_idx],
        "confidence": float(top5_probs[0].item())
    }
    
    # Top-5
    top5 = []
    for prob, idx in zip(top5_probs, top5_indices):
        top5.append({
            "label": classes[idx.item()],
            "confidence": float(prob.item())
        })
    
    return top1, top5

