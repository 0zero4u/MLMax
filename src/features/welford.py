
import numpy as np

class WelfordStats:
    """
    Implements Welford's online algorithm for computing mean and variance.
    This allows for stable, single-pass calculation of z-scores.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float('inf')
        self.max = float('-inf')

    def update(self, x):
        """Add a new value to the running statistics."""
        if x is None or np.isnan(x):
            return
            
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    @property
    def variance(self):
        """Returns the sample variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self):
        """Returns the sample standard deviation."""
        return np.sqrt(self.variance)

    def normalize(self, x):
        """Returns the z-score of x based on the running statistics."""
        if self.std < 1e-9:
            return 0.0
        return (x - self.mean) / self.std
