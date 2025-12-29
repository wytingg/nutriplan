"""
Logging utilities for NutriPlan experiments
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json


def setup_logger(
    name: str,
    log_file: str = None,
    level=logging.INFO,
    format_str: str = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_str: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Logger for tracking experiments with metadata"""

    def __init__(self, experiment_dir: str, experiment_name: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.log_dir = self.experiment_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Setup logger
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file)
        )

        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': timestamp,
            'log_file': str(log_file)
        }

        self.logger.info(f"Experiment logger initialized: {experiment_name}")

    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.metadata['config'] = config
        self.logger.info("="*80)
        self.logger.info("Experiment Configuration:")
        self.logger.info(json.dumps(config, indent=2))
        self.logger.info("="*80)

    def log_metrics(self, metrics: dict, step: int = None, prefix: str = ""):
        """Log metrics"""
        log_str = f"{prefix} Metrics" if prefix else "Metrics"
        if step is not None:
            log_str += f" (Step {step})"
        log_str += ":"

        self.logger.info(log_str)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_checkpoint(self, checkpoint_path: str, metrics: dict = None):
        """Log checkpoint save"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if metrics:
            self.logger.info(f"  Metrics: {metrics}")

    def log_error(self, error: Exception):
        """Log error"""
        self.logger.error(f"Error occurred: {type(error).__name__}: {str(error)}", exc_info=True)

    def finalize(self, final_metrics: dict = None):
        """Finalize experiment logging"""
        self.metadata['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics

        # Save metadata
        metadata_file = self.log_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info("="*80)
        self.logger.info("Experiment completed")
        if final_metrics:
            self.logger.info("Final metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        self.logger.info("="*80)


def log_gpu_info():
    """Log GPU information"""
    import torch
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        logger.info("="*80)
        logger.info("GPU Information:")
        logger.info(f"  CUDA Available: True")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

        logger.info("="*80)
    else:
        logger.info("CUDA not available. Using CPU.")


if __name__ == "__main__":
    # Test logger
    exp_logger = ExperimentLogger("test_logs", "test_experiment")

    config = {
        "model": "Llama-3",
        "learning_rate": 5e-5,
        "batch_size": 8
    }
    exp_logger.log_config(config)

    metrics = {
        "train_loss": 0.5234,
        "val_loss": 0.6123,
        "sncr": 0.8456
    }
    exp_logger.log_metrics(metrics, step=100)

    exp_logger.finalize(final_metrics=metrics)
