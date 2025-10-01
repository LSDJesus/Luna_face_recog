"""Surgical Vector Pruner for VAPOR-FACE MVP

Implements the core "Subtractive Semantics" functionality for systematically
nulling specific vector components and testing their semantic effects.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class PruningStrategy:
    """Base class for different pruning strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, vector: np.ndarray, indices: Union[slice, List[int], np.ndarray]) -> np.ndarray:
        """Apply pruning strategy to vector at specified indices"""
        pruned = vector.copy()
        pruned[indices] = 0.0
        return pruned


class ZeroPruning(PruningStrategy):
    """Set components to zero"""
    
    def __init__(self):
        super().__init__("zero")


class GaussianNoisePruning(PruningStrategy):
    """Replace components with Gaussian noise"""
    
    def __init__(self, std: float = 0.1, seed: Optional[int] = None):
        super().__init__("gaussian_noise")
        self.std = std
        self.seed = seed
    
    def apply(self, vector: np.ndarray, indices: Union[slice, List[int], np.ndarray]) -> np.ndarray:
        pruned = vector.copy()
        if self.seed is not None:
            np.random.seed(self.seed)
        noise = np.random.normal(0, self.std, size=np.sum(np.zeros_like(vector)[indices] == 0))
        pruned[indices] = noise
        return pruned


class MeanReplacementPruning(PruningStrategy):
    """Replace components with vector mean"""
    
    def __init__(self):
        super().__init__("mean_replacement")
    
    def apply(self, vector: np.ndarray, indices: Union[slice, List[int], np.ndarray]) -> np.ndarray:
        pruned = vector.copy()
        mean_val = np.mean(vector)
        pruned[indices] = mean_val
        return pruned


class SurgicalPruner:
    """
    Systematic vector pruning for semantic analysis
    
    Core component for the "Subtractive Semantics" experiments described
    in Document 4.2.C of the VAPOR-FACE specification.
    """
    
    def __init__(self):
        self.strategies = {
            "zero": ZeroPruning(),
            "gaussian": GaussianNoisePruning(),
            "mean": MeanReplacementPruning()
        }
        
        # Track pruning experiments
        self.experiment_history: List[Dict[str, Any]] = []
        
        logger.info("Surgical pruner initialized")
    
    def prune_axis(self, 
                   vector: np.ndarray,
                   axis_name: str,
                   axis_indices: Union[slice, List[int], np.ndarray],
                   strategy: str = "zero") -> Dict[str, Any]:
        """
        Prune a specific semantic axis
        
        Args:
            vector: Original semantic vector
            axis_name: Name of the semantic axis being pruned
            axis_indices: Indices or slice defining the axis
            strategy: Pruning strategy ("zero", "gaussian", "mean")
            
        Returns:
            Pruning result with original and pruned vectors
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
        
        try:
            # Apply pruning
            pruned_vector = self.strategies[strategy].apply(vector, axis_indices)
            
            # Calculate impact metrics
            impact = self._calculate_impact(vector, pruned_vector, axis_indices)
            
            result = {
                "axis_name": axis_name,
                "strategy": strategy,
                "original_vector": vector.copy(),
                "pruned_vector": pruned_vector,
                "axis_indices": axis_indices,
                "impact_metrics": impact,
                "success": True
            }
            
            # Add to experiment history
            self.experiment_history.append({
                "axis_name": axis_name,
                "strategy": strategy,
                "impact_metrics": impact,
                "timestamp": np.datetime64('now')
            })
            
            logger.info(f"Pruned axis '{axis_name}' using {strategy} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Pruning failed for axis '{axis_name}': {e}")
            return {
                "axis_name": axis_name,
                "strategy": strategy,
                "success": False,
                "error": str(e)
            }
    
    def prune_multiple_axes(self,
                           vector: np.ndarray,
                           axes_config: Dict[str, Union[slice, List[int], np.ndarray]],
                           strategy: str = "zero") -> Dict[str, Any]:
        """
        Prune multiple semantic axes simultaneously
        
        Args:
            vector: Original semantic vector
            axes_config: Dict mapping axis names to their indices
            strategy: Pruning strategy
            
        Returns:
            Combined pruning results
        """
        results = {}
        cumulative_pruned = vector.copy()
        
        for axis_name, axis_indices in axes_config.items():
            # Apply pruning to cumulative result
            result = self.prune_axis(cumulative_pruned, axis_name, axis_indices, strategy)
            if result["success"]:
                cumulative_pruned = result["pruned_vector"]
                results[axis_name] = result
            else:
                logger.warning(f"Failed to prune axis {axis_name}")
        
        return {
            "individual_results": results,
            "final_pruned_vector": cumulative_pruned,
            "original_vector": vector,
            "axes_pruned": list(axes_config.keys()),
            "strategy": strategy
        }
    
    def systematic_axis_scan(self,
                            vector: np.ndarray,
                            semantic_axes: Dict[str, Union[slice, List[int], np.ndarray]],
                            strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Systematically prune each semantic axis individually
        
        This implements the core "Semantic Archaeology" experiment from Document 4.2.C
        
        Args:
            vector: Original semantic vector
            semantic_axes: Dict mapping axis names to indices
            strategies: List of pruning strategies to test
            
        Returns:
            Complete scan results for all axes and strategies
        """
        if strategies is None:
            strategies = ["zero"]
        
        scan_results = {}
        
        for strategy in strategies:
            scan_results[strategy] = {}
            
            for axis_name, axis_indices in semantic_axes.items():
                result = self.prune_axis(vector, axis_name, axis_indices, strategy)
                scan_results[strategy][axis_name] = result
                
        logger.info(f"Systematic scan completed: {len(semantic_axes)} axes, {len(strategies)} strategies")
        return scan_results
    
    def _calculate_impact(self, 
                         original: np.ndarray, 
                         pruned: np.ndarray, 
                         indices: Union[slice, List[int], np.ndarray]) -> Dict[str, float]:
        """Calculate the impact of pruning"""
        
        # Vector distance metrics
        l2_distance = float(np.linalg.norm(original - pruned))
        
        # Avoid division by zero
        orig_norm = np.linalg.norm(original)
        pruned_norm = np.linalg.norm(pruned)
        
        if orig_norm > 0 and pruned_norm > 0:
            cosine_sim = float(np.dot(original, pruned) / (orig_norm * pruned_norm))
        else:
            cosine_sim = 0.0
        
        # Affected component statistics
        if isinstance(indices, slice):
            affected_components = original[indices]
        else:
            affected_components = original[indices]
        
        affected_mean = float(np.mean(affected_components))
        affected_std = float(np.std(affected_components))
        affected_count = len(affected_components)
        
        relative_impact = l2_distance / orig_norm if orig_norm > 0 else 0.0
        
        return {
            "l2_distance": l2_distance,
            "cosine_similarity": cosine_sim,
            "affected_count": affected_count,
            "affected_mean": affected_mean,
            "affected_std": affected_std,
            "relative_impact": float(relative_impact)
        }
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all pruning experiments"""
        if not self.experiment_history:
            return {"total_experiments": 0}
        
        # Aggregate statistics
        axes_tested = set(exp["axis_name"] for exp in self.experiment_history)
        strategies_used = set(exp["strategy"] for exp in self.experiment_history)
        
        avg_impact = np.mean([exp["impact_metrics"]["relative_impact"] 
                            for exp in self.experiment_history])
        
        return {
            "total_experiments": len(self.experiment_history),
            "unique_axes_tested": len(axes_tested),
            "strategies_used": list(strategies_used),
            "average_relative_impact": float(avg_impact),
            "axes_tested": list(axes_tested)
        }
    
    def clear_history(self):
        """Clear experiment history"""
        self.experiment_history.clear()
        logger.info("Pruning experiment history cleared")