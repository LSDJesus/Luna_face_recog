"""Main Application for VAPOR-FACE MVP

Simple GUI application for semantic facial recognition research.
Focused on surgical vector pruning experiments.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our core components
from core.face_extractor import FaceExtractor
from core.semantic_processor import SemanticProcessor
from core.vector_store import VectorStore

# Fix the surgical pruner import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.surgical_pruner_fixed import SurgicalPruner


class VaporFaceGUI:
    """Main GUI application for VAPOR-FACE MVP"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("VAPOR-FACE MVP - Semantic Facial Recognition Research")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.face_extractor = None
        self.semantic_processor = None
        self.vector_store = None
        self.surgical_pruner = None
        
        # Current data
        self.current_image_path = None
        self.current_face_crop = None
        self.current_embedding = None
        self.current_embedding_id = None
        self.current_semantic_axes = None
        
        # Setup GUI
        self.setup_gui()
        self.setup_components()
        
        logger.info("VAPOR-FACE MVP GUI initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frames
        self.setup_menu()
        self.setup_toolbar()
        self.setup_main_area()
        self.setup_status_bar()
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Database Statistics", command=self.show_db_stats)
        tools_menu.add_command(label="Clear Experiment History", command=self.clear_history)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Process Face", command=self.process_face).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Run Systematic Scan", command=self.run_systematic_scan).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(toolbar, text="Encoder:").pack(side=tk.LEFT, padx=2)
        self.encoder_var = tk.StringVar(value="mock")
        encoder_combo = ttk.Combobox(toolbar, textvariable=self.encoder_var, 
                                   values=["mock"], state="readonly", width=10)
        encoder_combo.pack(side=tk.LEFT, padx=2)\n    \n    def setup_main_area(self):\n        \"\"\"Setup main content area\"\"\"\n        # Create notebook for tabs\n        self.notebook = ttk.Notebook(self.root)\n        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)\n        \n        # Image processing tab\n        self.setup_image_tab()\n        \n        # Pruning experiments tab\n        self.setup_pruning_tab()\n        \n        # Results analysis tab\n        self.setup_results_tab()\n    \n    def setup_image_tab(self):\n        \"\"\"Setup image processing tab\"\"\"\n        image_frame = ttk.Frame(self.notebook)\n        self.notebook.add(image_frame, text="Image Processing")\n        \n        # Left panel for image display\n        left_panel = ttk.Frame(image_frame)\n        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)\n        \n        ttk.Label(left_panel, text="Original Image").pack()\n        self.original_image_label = ttk.Label(left_panel, text="No image loaded")\n        self.original_image_label.pack(pady=5)\n        \n        ttk.Label(left_panel, text="Extracted Face").pack()\n        self.face_crop_label = ttk.Label(left_panel, text="No face extracted")\n        self.face_crop_label.pack(pady=5)\n        \n        # Right panel for processing info\n        right_panel = ttk.Frame(image_frame)\n        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)\n        \n        ttk.Label(right_panel, text="Processing Information").pack()\n        self.processing_text = scrolledtext.ScrolledText(right_panel, width=40, height=20)\n        self.processing_text.pack(fill=tk.BOTH, expand=True)\n    \n    def setup_pruning_tab(self):\n        \"\"\"Setup pruning experiments tab\"\"\"\n        pruning_frame = ttk.Frame(self.notebook)\n        self.notebook.add(pruning_frame, text="Surgical Pruning")\n        \n        # Control panel\n        control_panel = ttk.LabelFrame(pruning_frame, text="Pruning Controls")\n        control_panel.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Axis selection\n        ttk.Label(control_panel, text="Semantic Axis:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)\n        self.axis_var = tk.StringVar()\n        self.axis_combo = ttk.Combobox(control_panel, textvariable=self.axis_var, state="readonly")\n        self.axis_combo.grid(row=0, column=1, padx=5, pady=2)\n        \n        # Strategy selection\n        ttk.Label(control_panel, text="Strategy:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)\n        self.strategy_var = tk.StringVar(value="zero")\n        strategy_combo = ttk.Combobox(control_panel, textvariable=self.strategy_var,\n                                    values=["zero", "gaussian", "mean"], state="readonly")\n        strategy_combo.grid(row=0, column=3, padx=5, pady=2)\n        \n        # Prune button\n        ttk.Button(control_panel, text="Prune Axis", command=self.prune_axis).grid(row=0, column=4, padx=5, pady=2)\n        \n        # Results display\n        results_panel = ttk.LabelFrame(pruning_frame, text="Pruning Results")\n        results_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)\n        \n        self.pruning_results_text = scrolledtext.ScrolledText(results_panel, height=15)\n        self.pruning_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)\n    \n    def setup_results_tab(self):\n        \"\"\"Setup results analysis tab\"\"\"\n        results_frame = ttk.Frame(self.notebook)\n        self.notebook.add(results_frame, text="Results Analysis")\n        \n        # Create matplotlib figure\n        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))\n        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)\n        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)\n        \n        # Control panel for plotting\n        plot_control = ttk.Frame(results_frame)\n        plot_control.pack(fill=tk.X, padx=5, pady=5)\n        \n        ttk.Button(plot_control, text="Plot Vector Comparison", command=self.plot_vector_comparison).pack(side=tk.LEFT, padx=5)\n        ttk.Button(plot_control, text="Plot Impact Analysis", command=self.plot_impact_analysis).pack(side=tk.LEFT, padx=5)\n    \n    def setup_status_bar(self):\n        \"\"\"Setup status bar\"\"\"\n        self.status_var = tk.StringVar(value="Ready")\n        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)\n        status_bar.pack(side=tk.BOTTOM, fill=tk.X)\n    \n    def setup_components(self):\n        \"\"\"Initialize processing components\"\"\"\n        try:\n            self.face_extractor = FaceExtractor()\n            self.semantic_processor = SemanticProcessor(encoder_type=self.encoder_var.get())\n            self.vector_store = VectorStore()\n            self.surgical_pruner = SurgicalPruner()\n            \n            self.update_status("Components initialized successfully")\n            \n        except Exception as e:\n            self.update_status(f"Component initialization failed: {e}")\n            logger.error(f"Component setup failed: {e}")\n    \n    def update_status(self, message: str):\n        \"\"\"Update status bar\"\"\"\n        self.status_var.set(message)\n        self.root.update_idletasks()\n        logger.info(message)\n    \n    def load_image(self):\n        \"\"\"Load and display an image\"\"\"\n        file_path = filedialog.askopenfilename(\n            title="Select Image",\n            filetypes=[(\"Image files\", \"*.jpg *.jpeg *.png *.bmp *.tiff\")]\n        )\n        \n        if not file_path:\n            return\n        \n        try:\n            self.current_image_path = file_path\n            \n            # Load and display original image\n            img = Image.open(file_path)\n            img.thumbnail((300, 300))\n            img_tk = ImageTk.PhotoImage(img)\n            \n            self.original_image_label.configure(image=img_tk, text=\"\")\n            self.original_image_label.image = img_tk  # Keep reference\n            \n            self.update_status(f\"Loaded image: {Path(file_path).name}\")\n            \n            # Clear previous results\n            self.current_face_crop = None\n            self.current_embedding = None\n            self.current_embedding_id = None\n            self.face_crop_label.configure(image=\"\", text=\"No face extracted\")\n            \n        except Exception as e:\n            messagebox.showerror(\"Error\", f\"Failed to load image: {e}\")\n            logger.error(f\"Image loading failed: {e}\")\n    \n    def process_face(self):\n        \"\"\"Extract face and generate semantic embedding\"\"\"\n        if not self.current_image_path:\n            messagebox.showwarning(\"Warning\", \"Please load an image first\")\n            return\n        \n        try:\n            self.update_status(\"Extracting face...\")\n            \n            # Extract face\n            face_crop = self.face_extractor.extract_face(self.current_image_path)\n            if face_crop is None:\n                messagebox.showerror(\"Error\", \"No face detected in image\")\n                return\n            \n            self.current_face_crop = face_crop\n            \n            # Display face crop\n            face_pil = Image.fromarray(face_crop)\n            face_pil.thumbnail((200, 200))\n            face_tk = ImageTk.PhotoImage(face_pil)\n            \n            self.face_crop_label.configure(image=face_tk, text=\"\")\n            self.face_crop_label.image = face_tk\n            \n            self.update_status(\"Generating semantic embedding...\")\n            \n            # Process with semantic encoder\n            result = self.semantic_processor.process_image(face_crop)\n            \n            if not result[\"success\"]:\n                messagebox.showerror(\"Error\", f\"Semantic processing failed: {result.get('error', 'Unknown error')}\")\n                return\n            \n            self.current_embedding = result[\"embedding\"]\n            \n            # Store in database\n            embedding_id = self.vector_store.store_embedding(\n                image_path=self.current_image_path,\n                embedding=self.current_embedding,\n                model_info=result[\"model_info\"],\n                processing_stats=result[\"statistics\"],\n                semantic_axes=result.get(\"semantic_axes\")\n            )\n            \n            self.current_embedding_id = embedding_id\n            self.current_semantic_axes = result.get(\"semantic_axes\", {})\n            \n            # Update axis combo box\n            if self.current_semantic_axes:\n                self.axis_combo[\"values\"] = list(self.current_semantic_axes.keys())\n                if self.axis_combo[\"values\"]:\n                    self.axis_combo.current(0)\n            \n            # Display processing info\n            info_text = f\"\"\"Face Processing Results:\n\nImage: {Path(self.current_image_path).name}\nFace Size: {face_crop.shape}\nEmbedding Dimension: {len(self.current_embedding)}\nDatabase ID: {embedding_id}\n\nModel Info:\n{json.dumps(result['model_info'], indent=2)}\n\nStatistics:\n{json.dumps(result['statistics'], indent=2)}\n\nSemantic Axes Available: {len(self.current_semantic_axes)}\n\"\"\"\n            \n            self.processing_text.delete(1.0, tk.END)\n            self.processing_text.insert(1.0, info_text)\n            \n            self.update_status(\"Face processing completed successfully\")\n            \n        except Exception as e:\n            messagebox.showerror(\"Error\", f\"Face processing failed: {e}\")\n            logger.error(f\"Face processing failed: {e}\")\n    \n    def prune_axis(self):\n        \"\"\"Prune selected semantic axis\"\"\"\n        if self.current_embedding is None:\n            messagebox.showwarning(\"Warning\", \"Please process a face first\")\n            return\n        \n        axis_name = self.axis_var.get()\n        if not axis_name:\n            messagebox.showwarning(\"Warning\", \"Please select a semantic axis\")\n            return\n        \n        try:\n            strategy = self.strategy_var.get()\n            axis_indices = self.current_semantic_axes[axis_name]\n            \n            self.update_status(f\"Pruning axis '{axis_name}' with {strategy} strategy...\")\n            \n            # Perform pruning\n            result = self.surgical_pruner.prune_axis(\n                vector=self.current_embedding,\n                axis_name=axis_name,\n                axis_indices=axis_indices,\n                strategy=strategy\n            )\n            \n            if not result[\"success\"]:\n                messagebox.showerror(\"Error\", f\"Pruning failed: {result.get('error', 'Unknown error')}\")\n                return\n            \n            # Store experiment in database\n            if self.current_embedding_id:\n                self.vector_store.store_pruning_experiment(\n                    embedding_id=self.current_embedding_id,\n                    axis_name=axis_name,\n                    strategy=strategy,\n                    pruned_embedding=result[\"pruned_vector\"],\n                    impact_metrics=result[\"impact_metrics\"]\n                )\n            \n            # Display results\n            results_text = f\"\"\"Pruning Results for '{axis_name}' (Strategy: {strategy}):\n\nImpact Metrics:\n{json.dumps(result['impact_metrics'], indent=2)}\n\nVector Changes:\n- Original norm: {np.linalg.norm(result['original_vector']):.4f}\n- Pruned norm: {np.linalg.norm(result['pruned_vector']):.4f}\n- L2 distance: {result['impact_metrics']['l2_distance']:.4f}\n- Cosine similarity: {result['impact_metrics']['cosine_similarity']:.4f}\n\n{'-'*50}\n\"\"\"\n            \n            self.pruning_results_text.insert(tk.END, results_text)\n            self.pruning_results_text.see(tk.END)\n            \n            self.update_status(f\"Pruning completed: {axis_name} ({strategy})\")\n            \n        except Exception as e:\n            messagebox.showerror(\"Error\", f\"Pruning failed: {e}\")\n            logger.error(f\"Pruning failed: {e}\")\n    \n    def run_systematic_scan(self):\n        \"\"\"Run systematic pruning scan on all semantic axes\"\"\"\n        if self.current_embedding is None:\n            messagebox.showwarning(\"Warning\", \"Please process a face first\")\n            return\n        \n        if not self.current_semantic_axes:\n            messagebox.showwarning(\"Warning\", \"No semantic axes available\")\n            return\n        \n        try:\n            self.update_status(\"Running systematic pruning scan...\")\n            \n            # Run systematic scan\n            scan_results = self.surgical_pruner.systematic_axis_scan(\n                vector=self.current_embedding,\n                semantic_axes=self.current_semantic_axes,\n                strategies=[\"zero\", \"gaussian\", \"mean\"]\n            )\n            \n            # Display summary\n            summary_text = \"\\n\\nSYSTEMATIC PRUNING SCAN RESULTS:\\n\" + \"=\"*50 + \"\\n\"\n            \n            for strategy, results in scan_results.items():\n                summary_text += f\"\\nStrategy: {strategy.upper()}\\n\" + \"-\"*30 + \"\\n\"\n                \n                for axis_name, result in results.items():\n                    if result[\"success\"]:\n                        impact = result[\"impact_metrics\"]\n                        summary_text += f\"{axis_name:15} | L2: {impact['l2_distance']:.4f} | Cos: {impact['cosine_similarity']:.4f} | Impact: {impact['relative_impact']:.4f}\\n\"\n                    else:\n                        summary_text += f\"{axis_name:15} | FAILED: {result.get('error', 'Unknown')}\\n\"\n            \n            # Get experiment summary\n            exp_summary = self.surgical_pruner.get_experiment_summary()\n            summary_text += f\"\\n\\nExperiment Summary:\\n{json.dumps(exp_summary, indent=2)}\\n\"\n            \n            self.pruning_results_text.insert(tk.END, summary_text)\n            self.pruning_results_text.see(tk.END)\n            \n            self.update_status(\"Systematic scan completed\")\n            \n        except Exception as e:\n            messagebox.showerror(\"Error\", f\"Systematic scan failed: {e}\")\n            logger.error(f\"Systematic scan failed: {e}\")\n    \n    def plot_vector_comparison(self):\n        \"\"\"Plot comparison between original and pruned vectors\"\"\"\n        # Implementation for vector visualization\n        self.ax1.clear()\n        self.ax1.set_title(\"Vector Comparison (Placeholder)\")\n        self.ax1.text(0.5, 0.5, \"Vector comparison plot\\n(Implementation needed)\", \n                     ha=\"center\", va=\"center\", transform=self.ax1.transAxes)\n        self.canvas.draw()\n    \n    def plot_impact_analysis(self):\n        \"\"\"Plot impact analysis of pruning experiments\"\"\"\n        # Implementation for impact visualization\n        self.ax2.clear()\n        self.ax2.set_title(\"Impact Analysis (Placeholder)\")\n        self.ax2.text(0.5, 0.5, \"Impact analysis plot\\n(Implementation needed)\", \n                     ha=\"center\", va=\"center\", transform=self.ax2.transAxes)\n        self.canvas.draw()\n    \n    def show_db_stats(self):\n        \"\"\"Show database statistics\"\"\"\n        try:\n            stats = self.vector_store.get_statistics()\n            stats_text = f\"\"\"Database Statistics:\n\n{json.dumps(stats, indent=2)}\"\"\"\n            \n            messagebox.showinfo(\"Database Statistics\", stats_text)\n            \n        except Exception as e:\n            messagebox.showerror(\"Error\", f\"Failed to get statistics: {e}\")\n    \n    def clear_history(self):\n        \"\"\"Clear experiment history\"\"\"\n        if messagebox.askyesno(\"Confirm\", \"Clear all experiment history?\"):\n            self.surgical_pruner.clear_history()\n            self.pruning_results_text.delete(1.0, tk.END)\n            self.update_status(\"Experiment history cleared\")\n    \n    def export_results(self):\n        \"\"\"Export results to file\"\"\"\n        # Implementation for exporting results\n        messagebox.showinfo(\"Export\", \"Export functionality not yet implemented\")\n    \n    def show_about(self):\n        \"\"\"Show about dialog\"\"\"\n        about_text = \"\"\"VAPOR-FACE MVP v0.1.0\n\nSemantic Facial Recognition Research Platform\n\nMinimum Viable Proof-of-Concept for the\nSovereign Persona Archive project.\n\nFocused on surgical vector pruning and\nsemantic archaeology experiments.\"\"\"\n        \n        messagebox.showinfo(\"About VAPOR-FACE MVP\", about_text)\n\n\ndef main():\n    \"\"\"Main entry point\"\"\"\n    root = tk.Tk()\n    app = VaporFaceGUI(root)\n    root.mainloop()\n\n\nif __name__ == \"__main__\":\n    main()