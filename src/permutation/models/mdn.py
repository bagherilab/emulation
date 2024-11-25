from typing import Optional, Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import TransformerMixin
from sklearn.metrics import r2_score

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import HParams


class MDN(nn.Module):
    def __init__(self, hidden_dim=1, output_dim=1, num_gaussians=1, learning_rate=0.001, epochs=1000):
        super(MDN, self).__init__()
        self.hidden = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi_layer = nn.Linear(hidden_dim, num_gaussians)  # Mixing coefficients
        self.mu_layer = nn.Linear(hidden_dim, num_gaussians * output_dim)  # Means
        self.sigma_layer = nn.Linear(hidden_dim, num_gaussians * output_dim)  # Std deviations
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.epochs = epochs

    def forward(self, x):
        hidden = self.hidden(x)
        pi = torch.softmax(self.pi_layer(hidden), dim=1)  # Mixing coefficients
        mu = self.mu_layer(hidden).view(-1, self.num_gaussians, self.output_dim)  # Means
        sigma = torch.exp(self.sigma_layer(hidden)).view(-1, self.num_gaussians, self.output_dim)  # Variances
        return pi, mu, sigma
    
    def _mdn_loss(self, pi, mu, sigma, y):
        """MDN negative log-likelihood loss"""
        y = y.unsqueeze(1) if len(y.shape) == 2 else y.unsqueeze(1).unsqueeze(-1)
        gaussian_prob = torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        weighted_prob = pi * torch.prod(gaussian_prob, dim=2)
        loss = -torch.log(torch.sum(weighted_prob, dim=1) + 1e-6).mean()  # Avoid log(0) with 1e-6
        return loss


    def fit(self, X, y):
        """Train the MDN using PyTorch."""
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.to_numpy(), dtype=torch.float32)

        for _ in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            pi, mu, sigma = self.forward(X)
            loss = self._mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer.step()

        return self
    
    def predict(self, X):
        """Predict using the trained MDN."""
        X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self.forward(X)
            pi = pi.unsqueeze(-1)
            predictions = torch.sum(pi * mu, dim=1)
        return predictions.numpy()
    
    def score(self, X, y, sample_weight=None):
        """Compute the R^2 score (default for regression in scikit-learn)."""
        predictions = self.predict(X)  # Call the predict method
        return r2_score(y, predictions, sample_weight=sample_weight)  # Use R^2 as the scoring metric
    

class MDNReg(AbstractSKLearnModel):
    """
    Mixture Density Network Regressor compatible with scikit-learn.
    """

    algorithm_name = "Mixture Density Network Regressor"
    algorithm_abv = "MDN"
    algorithm_type = "Regression"

    def __init__(self):
        """
        """
        self.model = None

        
    @classmethod
    def set_model(
        cls,
        model_dependency: nn.Module = MDN,
        hparams: Optional[HParams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """Set up model from config files and superclass."""
        if preprocessing_dependencies is None:
            preprocessing_dependencies = []
        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )


    