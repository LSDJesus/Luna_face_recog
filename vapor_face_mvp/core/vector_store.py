"""Vector Database for VAPOR-FACE MVP

Handles storage and retrieval of semantic embeddings and associated metadata.
Uses SQLite for simplicity in the MVP phase.
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """
    SQLite-based vector storage for semantic embeddings
    
    Stores:
    - Original images (file paths)
    - Face crops (as binary blobs or file paths)
    - Semantic embeddings (as JSON arrays)
    - Processing metadata
    - Pruning experiment results
    """
    
    def __init__(self, db_path: str = "vapor_face_mvp.db"):
        """
        Initialize vector database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = None
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Vector store initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.connection.cursor()
        
        # Main embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                face_crop_path TEXT,
                embedding_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                model_info TEXT,
                processing_stats TEXT,
                semantic_axes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_path)
            )
        """)
        
        # Pruning experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pruning_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_id INTEGER NOT NULL,
                experiment_name TEXT,
                axis_name TEXT NOT NULL,
                strategy TEXT NOT NULL,
                pruned_embedding_json TEXT NOT NULL,
                impact_metrics TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (embedding_id) REFERENCES embeddings (id)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_path ON embeddings(image_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_id ON pruning_experiments(embedding_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_axis_name ON pruning_experiments(axis_name)")
        
        self.connection.commit()
        logger.info("Database schema initialized")
    
    def store_embedding(self,
                       image_path: str,
                       embedding: np.ndarray,
                       face_crop_path: Optional[str] = None,
                       model_info: Optional[Dict[str, Any]] = None,
                       processing_stats: Optional[Dict[str, Any]] = None,
                       semantic_axes: Optional[Dict[str, Any]] = None) -> int:
        """
        Store semantic embedding and metadata
        
        Args:
            image_path: Path to original image
            embedding: Semantic embedding vector
            face_crop_path: Path to extracted face crop
            model_info: Information about the model used
            processing_stats: Processing statistics
            semantic_axes: Semantic axis mappings
            
        Returns:
            Database ID of stored embedding
        """
        cursor = self.connection.cursor()
        
        try:
            # Convert numpy array to JSON
            embedding_json = json.dumps(embedding.tolist())
            
            # Convert metadata to JSON with proper serialization for slice objects
            model_info_json = json.dumps(model_info) if model_info else None
            processing_stats_json = json.dumps(processing_stats) if processing_stats else None
            
            # Convert semantic axes to JSON-serializable format
            if semantic_axes:
                serializable_axes = {}
                for name, indices in semantic_axes.items():
                    if isinstance(indices, slice):
                        # Convert slice to dict
                        serializable_axes[name] = {
                            "type": "slice",
                            "start": indices.start,
                            "stop": indices.stop,
                            "step": indices.step
                        }
                    elif isinstance(indices, (list, np.ndarray)):
                        # Convert array/list to list
                        serializable_axes[name] = {
                            "type": "indices",
                            "values": list(indices)
                        }
                    else:
                        # Other types - convert to string representation
                        serializable_axes[name] = {
                            "type": "other",
                            "value": str(indices)
                        }
                semantic_axes_json = json.dumps(serializable_axes)
            else:
                semantic_axes_json = None
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (image_path, face_crop_path, embedding_json, dimension, 
                 model_info, processing_stats, semantic_axes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                image_path,
                face_crop_path,
                embedding_json,
                len(embedding),
                model_info_json,
                processing_stats_json,
                semantic_axes_json
            ))
            
            embedding_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Stored embedding for {image_path} (ID: {embedding_id})")
            return embedding_id
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to store embedding: {e}")
            raise
    
    def get_embedding(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding by ID
        
        Args:
            embedding_id: Database ID
            
        Returns:
            Embedding data or None if not found
        """
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT * FROM embeddings WHERE id = ?", (embedding_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Parse JSON fields
        embedding = np.array(json.loads(row["embedding_json"]))
        model_info = json.loads(row["model_info"]) if row["model_info"] else None
        processing_stats = json.loads(row["processing_stats"]) if row["processing_stats"] else None
        semantic_axes = json.loads(row["semantic_axes"]) if row["semantic_axes"] else None
        
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "face_crop_path": row["face_crop_path"],
            "embedding": embedding,
            "dimension": row["dimension"],
            "model_info": model_info,
            "processing_stats": processing_stats,
            "semantic_axes": semantic_axes,
            "created_at": row["created_at"]
        }
    
    def get_embedding_by_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding by image path
        
        Args:
            image_path: Path to original image
            
        Returns:
            Embedding data or None if not found
        """
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT * FROM embeddings WHERE image_path = ?", (image_path,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self.get_embedding(row["id"])
    
    def store_pruning_experiment(self,
                                embedding_id: int,
                                axis_name: str,
                                strategy: str,
                                pruned_embedding: np.ndarray,
                                impact_metrics: Dict[str, Any],
                                experiment_name: Optional[str] = None) -> int:
        """
        Store pruning experiment results
        
        Args:
            embedding_id: ID of original embedding
            axis_name: Name of pruned axis
            strategy: Pruning strategy used
            pruned_embedding: Resulting pruned vector
            impact_metrics: Impact analysis results
            experiment_name: Optional experiment identifier
            
        Returns:
            Database ID of stored experiment
        """
        cursor = self.connection.cursor()
        
        try:
            # Convert to JSON with proper float conversion
            pruned_embedding_json = json.dumps(pruned_embedding.tolist())
            
            # Convert numpy types to regular Python types for JSON serialization
            serializable_metrics = {}
            for key, value in impact_metrics.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            impact_metrics_json = json.dumps(serializable_metrics)
            
            cursor.execute("""
                INSERT INTO pruning_experiments 
                (embedding_id, experiment_name, axis_name, strategy, 
                 pruned_embedding_json, impact_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                embedding_id,
                experiment_name,
                axis_name,
                strategy,
                pruned_embedding_json,
                impact_metrics_json
            ))
            
            experiment_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Stored pruning experiment: {axis_name} ({strategy}) for embedding {embedding_id}")
            return experiment_id
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to store pruning experiment: {e}")
            raise
    
    def get_pruning_experiments(self, embedding_id: int) -> List[Dict[str, Any]]:
        """
        Get all pruning experiments for an embedding
        
        Args:
            embedding_id: Original embedding ID
            
        Returns:
            List of experiment records
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT * FROM pruning_experiments 
            WHERE embedding_id = ? 
            ORDER BY created_at DESC
        """, (embedding_id,))
        
        experiments = []
        for row in cursor.fetchall():
            pruned_embedding = np.array(json.loads(row["pruned_embedding_json"]))
            impact_metrics = json.loads(row["impact_metrics"])
            
            experiments.append({
                "id": row["id"],
                "experiment_name": row["experiment_name"],
                "axis_name": row["axis_name"],
                "strategy": row["strategy"],
                "pruned_embedding": pruned_embedding,
                "impact_metrics": impact_metrics,
                "created_at": row["created_at"]
            })
        
        return experiments
    
    def list_embeddings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all stored embeddings (metadata only)
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of embedding metadata
        """
        cursor = self.connection.cursor()
        
        query = """
            SELECT id, image_path, face_crop_path, dimension, created_at 
            FROM embeddings 
            ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.connection.cursor()
        
        # Count embeddings
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        embedding_count = cursor.fetchone()["count"]
        
        # Count experiments
        cursor.execute("SELECT COUNT(*) as count FROM pruning_experiments")
        experiment_count = cursor.fetchone()["count"]
        
        # Count unique axes tested
        cursor.execute("SELECT COUNT(DISTINCT axis_name) as count FROM pruning_experiments")
        unique_axes = cursor.fetchone()["count"]
        
        return {
            "total_embeddings": embedding_count,
            "total_experiments": experiment_count,
            "unique_axes_tested": unique_axes,
            "database_path": str(self.db_path)
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()