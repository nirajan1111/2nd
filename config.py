"""
Configuration Management for Legal Reasoning System
Centralized configuration for training, inference, and model parameters
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==============================================================================
# PATHS
# ==============================================================================

@dataclass
class Paths:
    """Directory and file paths"""
    # Data
    data_dir: Path = PROJECT_ROOT / "data"
    legal_clauses: Path = data_dir / "legal_clauses.json"
    predicates: Path = data_dir / "predicates.txt"
    
    # Checkpoints
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    t5_fopl_checkpoint: Path = checkpoints_dir / "t5_fopl"
    best_model: Path = checkpoints_dir / "best_model"
    
    # CUAD
    cuad_dir: Path = PROJECT_ROOT / "CUAD_v1"
    cuad_data: Path = cuad_dir / "CUAD_v1.json"
    cuad_models: Path = PROJECT_ROOT / "cuad_models"
    
    # Outputs
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    results_file: Path = PROJECT_ROOT / "reasoning_results.json"
    
    def ensure_dirs(self):
        """Create necessary directories"""
        for attr_name in dir(self):
            if attr_name.endswith('_dir'):
                dir_path = getattr(self, attr_name)
                dir_path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

@dataclass
class T5FOPLConfig:
    """T5 Model Configuration for FOPL Generation"""
    # Model
    model_name: str = "google/t5-v1_1-base"  # English-only T5
    max_input_length: int = 512
    max_output_length: int = 128
    
    # Training
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Optimization
    gradient_checkpointing: bool = False
    fp16: bool = False  # Set True if GPU available
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    save_total_limit: int = 2
    
    # Early stopping
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_loss"
    
    # Generation
    num_beams: int = 4
    do_sample: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ComplianceConfig:
    """Compliance Checker Configuration"""
    # CUAD Model for clause extraction
    cuad_model: str = "roberta-base"
    cuad_checkpoint: Optional[str] = None
    
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence: float = 0.85
    
    # Processing
    max_clause_length: int = 512
    batch_size: int = 16


@dataclass
class InferenceConfig:
    """Inference Pipeline Configuration"""
    # Models
    use_trained_fopl: bool = True  # Use trained T5 model
    fopl_model_path: Optional[str] = None
    
    # Compliance
    use_cuad: bool = True
    
    # Processing
    batch_size: int = 8
    max_clauses: int = 100
    
    # Output
    include_explanations: bool = True
    save_intermediate: bool = False


# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.paths = Paths()
        self.t5_fopl = T5FOPLConfig()
        self.compliance = ComplianceConfig()
        self.inference = InferenceConfig()
        
        # Ensure directories exist
        self.paths.ensure_dirs()
    
    def get_device(self) -> str:
        """Get compute device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def update_for_gpu(self):
        """Update config for GPU training"""
        device = self.get_device()
        if device in ["cuda", "mps"]:
            self.t5_fopl.fp16 = (device == "cuda")  # FP16 only for CUDA
            self.t5_fopl.batch_size = 16
            self.t5_fopl.eval_batch_size = 8
            print(f"✅ GPU detected ({device}). Enabled optimizations.")
        else:
            print(f"⚠️  No GPU detected. Training will be slower.")
    
    def update(self, **kwargs):
        """Update configuration from kwargs"""
        for key, value in kwargs.items():
            if hasattr(self.t5_fopl, key):
                setattr(self.t5_fopl, key, value)
            elif hasattr(self.compliance, key):
                setattr(self.compliance, key, value)
            elif hasattr(self.inference, key):
                setattr(self.inference, key, value)
    
    def save(self, path: str):
        """Save configuration to JSON"""
        import json
        config_dict = {
            't5_fopl': self.t5_fopl.to_dict(),
            'compliance': self.compliance.__dict__,
            'inference': self.inference.__dict__
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON"""
        import json
        config = cls()
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Update T5 config
        for key, value in config_dict.get('t5_fopl', {}).items():
            if hasattr(config.t5_fopl, key):
                setattr(config.t5_fopl, key, value)
        
        # Update Compliance config
        for key, value in config_dict.get('compliance', {}).items():
            if hasattr(config.compliance, key):
                setattr(config.compliance, key, value)
        
        # Update Inference config
        for key, value in config_dict.get('inference', {}).items():
            if hasattr(config.inference, key):
                setattr(config.inference, key, value)
        
        return config


# ==============================================================================
# GLOBAL CONFIG INSTANCE
# ==============================================================================

# Create default config
config = Config()

# Export commonly used paths
PATHS = config.paths
DATA_DIR = config.paths.data_dir
CHECKPOINTS_DIR = config.paths.checkpoints_dir
OUTPUTS_DIR = config.paths.outputs_dir


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get configuration instance
    
    Args:
        config_file: Path to JSON config file (optional)
    
    Returns:
        Config instance
    """
    if config_file and os.path.exists(config_file):
        return Config.load(config_file)
    return Config()


def print_config(cfg: Config):
    """Pretty print configuration"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    
    print("\n📁 Paths:")
    print(f"  Data: {cfg.paths.data_dir}")
    print(f"  Checkpoints: {cfg.paths.checkpoints_dir}")
    print(f"  Outputs: {cfg.paths.outputs_dir}")
    
    print("\n🤖 T5 FOPL Model:")
    print(f"  Model: {cfg.t5_fopl.model_name}")
    print(f"  Epochs: {cfg.t5_fopl.num_epochs}")
    print(f"  Batch Size: {cfg.t5_fopl.batch_size}")
    print(f"  Learning Rate: {cfg.t5_fopl.learning_rate}")
    print(f"  FP16: {cfg.t5_fopl.fp16}")
    
    print("\n✅ Compliance:")
    print(f"  CUAD Model: {cfg.compliance.cuad_model}")
    print(f"  Min Confidence: {cfg.compliance.min_confidence}")
    
    print("\n🔮 Inference:")
    print(f"  Use Trained FOPL: {cfg.inference.use_trained_fopl}")
    print(f"  Batch Size: {cfg.inference.batch_size}")
    
    print("\n💻 Device: " + cfg.get_device())
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration
    cfg = Config()
    cfg.update_for_gpu()
    print_config(cfg)
    
    # Save example config
    cfg.save("config_example.json")
    print("✅ Saved example config to config_example.json")
