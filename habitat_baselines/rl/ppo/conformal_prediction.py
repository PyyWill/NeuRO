import torch
import numpy as np
from collections import deque
from typing import Optional, List


class ConformalQuantile:
    """
    Conformal prediction for quantile threshold computation.
    """
    
    def __init__(self, alpha: float = 0.1, buffer_size: int = 1000):
        """
        Args:
            alpha: Risk level (1 - coverage level)
            buffer_size: Size of rolling buffer for calibration data
        """
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.calibration_scores = deque(maxlen=buffer_size)
        self.q_threshold = None
        
    def add_calibration_score(self, score: float):
        """Add a calibration score to the buffer."""
        self.calibration_scores.append(score)
        
    def compute_quantile_threshold(self) -> float:
        """Compute quantile threshold from calibration scores."""
        if len(self.calibration_scores) < 10:
            return 0.5
            
        scores = np.array(list(self.calibration_scores))
        q = np.quantile(scores, 1 - self.alpha)
        self.q_threshold = q
        return q
        
    def get_threshold(self) -> float:
        """Get current quantile threshold."""
        if self.q_threshold is None:
            return self.compute_quantile_threshold()
        return self.q_threshold


def compute_conformal_quantile(
    model, 
    x_batch: torch.Tensor, 
    y_batch: torch.Tensor, 
    alpha: float = 0.1
) -> float:
    """
    Compute conformal quantile threshold for a batch of data.
    
    Args:
        model: PICNN model
        x_batch: Input batch [batch_size, input_dim]
        y_batch: Target batch [batch_size, y_dim]
        alpha: Risk level
        
    Returns:
        q: Quantile threshold
    """
    with torch.no_grad():
        scores = model(x_batch, y_batch)
        scores_np = scores.cpu().numpy()
        
    q = np.quantile(scores_np, 1 - alpha)
    return q
