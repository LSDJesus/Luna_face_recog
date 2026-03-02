"""
Semantic Probe Training for VAPOR-FACE MVP
==========================================
Linear probes for discovering interpretable semantic axes in face recognition embeddings.

This module implements TCAV-style (Testing with Concept Activation Vectors) probe training
to extract semantic direction vectors from labeled datasets like CelebA.

Author: Brian & Luna
Created: 2025-10-22
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class ProbeResult:
    """Results from training a single semantic probe."""
    attribute: str
    direction: np.ndarray  # Normalized concept direction vector
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    n_samples: int
    positive_ratio: float  # Class balance
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            'attribute': self.attribute,
            'direction': self.direction.tolist(),
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'n_samples': int(self.n_samples),
            'positive_ratio': float(self.positive_ratio),
        }


class SemanticProbeTrainer:
    """
    Train linear probes to discover semantic axes in face embeddings.
    
    This is the heart of our semantic archaeology—we're training simple
    linear classifiers to predict attributes (e.g., "Smiling", "Male", "Eyeglasses")
    from FR embeddings, then extracting the learned weight vectors as semantic
    direction axes that can be surgically manipulated.
    
    The beauty: these probes are trained post-hoc on frozen FR embeddings,
    so we discover what semantic structure *already exists* in ArcFace space.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        max_iter: int = 1000,
        **kwargs
    ):
        """
        Initialize probe trainer.
        
        Args:
            random_state: Random seed for reproducibility
            test_size: Fraction of data for testing
            max_iter: Max iterations for logistic regression
            **kwargs: Additional LogisticRegression arguments
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.random_state = random_state
        self.test_size = test_size
        self.max_iter = max_iter
        self.lr_kwargs = kwargs
        
        # Storage for trained probes
        self.probes: Dict[str, ProbeResult] = {}
    
    def train_probe(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        attribute: str,
        verbose: bool = True
    ) -> ProbeResult:
        """
        Train a single linear probe for an attribute.
        
        Args:
            embeddings: Face embeddings (N, D) - typically (N, 512) for ArcFace
            labels: Binary labels (N,) - 0/1 for negative/positive attribute
            attribute: Attribute name (e.g., "Smiling", "Male")
            verbose: Print training progress
        
        Returns:
            ProbeResult with trained direction vector and metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        if verbose:
            pos_ratio = labels.mean()
            print(f"📊 Training probe for '{attribute}':")
            print(f"   Samples: {len(labels)} (train: {len(X_train)}, test: {len(X_test)})")
            print(f"   Class balance: {pos_ratio:.1%} positive, {1-pos_ratio:.1%} negative")
        
        # Train logistic regression probe
        probe = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.lr_kwargs
        )
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_pred = probe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"   ✅ Accuracy: {accuracy:.1%}")
            print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Extract and normalize concept direction vector
        # For binary classification, coef_ has shape (1, D)
        direction = probe.coef_[0]
        direction = direction / np.linalg.norm(direction)  # L2 normalize
        
        result = ProbeResult(
            attribute=attribute,
            direction=direction,
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            n_samples=len(labels),
            positive_ratio=float(labels.mean())
        )
        
        # Store probe
        self.probes[attribute] = result
        
        return result
    
    def train_all_probes(
        self,
        embeddings: np.ndarray,
        attributes: Dict[str, np.ndarray],
        verbose: bool = True
    ) -> Dict[str, ProbeResult]:
        """
        Train probes for multiple attributes in batch.
        
        Args:
            embeddings: Face embeddings (N, D)
            attributes: Dict mapping attribute names to binary labels (N,)
            verbose: Print progress
        
        Returns:
            Dict mapping attribute names to ProbeResults
        """
        if verbose:
            print(f"🔬 Training {len(attributes)} semantic probes...")
            print(f"   Embedding shape: {embeddings.shape}")
        
        results = {}
        for attr_name, labels in attributes.items():
            result = self.train_probe(embeddings, labels, attr_name, verbose=verbose)
            results[attr_name] = result
            if verbose:
                print()  # Blank line between attributes
        
        if verbose:
            print(f"✅ Trained {len(results)} probes successfully!")
        
        return results
    
    def get_semantic_axes(self) -> Dict[str, np.ndarray]:
        """
        Get all trained semantic direction vectors.
        
        Returns:
            Dict mapping attribute names to normalized direction vectors
        """
        return {attr: probe.direction for attr, probe in self.probes.items()}
    
    def save_probes(self, path: Path | str) -> None:
        """
        Save trained probes to JSON file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        data = {
            'probes': {attr: probe.to_dict() for attr, probe in self.probes.items()},
            'metadata': {
                'n_probes': len(self.probes),
                'random_state': self.random_state,
                'test_size': self.test_size,
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.probes)} probes to {path}")
    
    def load_probes(self, path: Path | str) -> None:
        """
        Load trained probes from JSON file.
        
        Args:
            path: Input file path
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.probes = {}
        for attr, probe_dict in data['probes'].items():
            self.probes[attr] = ProbeResult(
                attribute=probe_dict['attribute'],
                direction=np.array(probe_dict['direction']),
                accuracy=probe_dict['accuracy'],
                precision=probe_dict['precision'],
                recall=probe_dict['recall'],
                f1_score=probe_dict['f1_score'],
                n_samples=probe_dict['n_samples'],
                positive_ratio=probe_dict['positive_ratio'],
            )
        
        print(f"📂 Loaded {len(self.probes)} probes from {path}")
    
    def get_probe_summary(self) -> str:
        """Get summary statistics for all trained probes."""
        if not self.probes:
            return "No probes trained yet."
        
        lines = [
            f"{'Attribute':<20} {'Accuracy':<10} {'F1':<8} {'Samples':<10} {'Balance':<10}",
            "-" * 70
        ]
        
        for attr, probe in sorted(self.probes.items(), key=lambda x: x[1].accuracy, reverse=True):
            lines.append(
                f"{attr:<20} {probe.accuracy:>8.1%}  {probe.f1_score:>6.3f}  "
                f"{probe.n_samples:>8}  {probe.positive_ratio:>8.1%}"
            )
        
        avg_acc = np.mean([p.accuracy for p in self.probes.values()])
        lines.append("-" * 70)
        lines.append(f"Average accuracy: {avg_acc:.1%} across {len(self.probes)} probes")
        
        return "\n".join(lines)


# Convenience function for quick probe training
def train_probes_from_dataset(
    embeddings: np.ndarray,
    attributes: Dict[str, np.ndarray],
    save_path: Optional[Path | str] = None,
    **kwargs
) -> Dict[str, ProbeResult]:
    """
    Quick utility to train probes and optionally save results.
    
    Args:
        embeddings: Face embeddings (N, D)
        attributes: Dict of attribute names to binary labels
        save_path: Optional path to save trained probes
        **kwargs: Additional SemanticProbeTrainer arguments
    
    Returns:
        Dict of trained ProbeResults
    """
    trainer = SemanticProbeTrainer(**kwargs)
    results = trainer.train_all_probes(embeddings, attributes)
    
    if save_path:
        trainer.save_probes(save_path)
    
    return results
