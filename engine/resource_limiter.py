"""
Resource limiter for training sessions.

Applies CPU, GPU, and memory constraints to prevent system overload
during training. Uses conservative defaults (50% CPU cores, GPU pause
at 75% utilization) to keep the system responsive.

Usage:
    from engine.resource_limiter import apply_resource_limits, get_resource_config
    apply_resource_limits()  # Call once before training starts
"""

import logging
import os
from typing import Optional

import torch

from engine.config import ResourceConfig

logger = logging.getLogger(__name__)

# Singleton config instance
_resource_config: Optional[ResourceConfig] = None


def get_resource_config() -> ResourceConfig:
    """Get the current resource configuration (creates default if needed)."""
    global _resource_config
    if _resource_config is None:
        _resource_config = ResourceConfig()
    return _resource_config


def set_resource_config(config: ResourceConfig) -> None:
    """Override the resource configuration."""
    global _resource_config
    _resource_config = config


def apply_resource_limits(config: Optional[ResourceConfig] = None) -> None:
    """Apply resource limits before training starts.

    Sets PyTorch CPU thread limits and GPU memory constraints.
    Should be called once at the beginning of a training session.

    Args:
        config: Resource configuration. Uses default if None.
    """
    if config is None:
        config = get_resource_config()

    _apply_cpu_limits(config)
    _apply_gpu_limits(config)

    logger.info(
        f"Resource limits applied: "
        f"CPU threads={torch.get_num_threads()}, "
        f"interop threads={torch.get_num_interop_threads()}, "
        f"GPU memory fraction={config.gpu_max_memory_fraction}"
    )


def _apply_cpu_limits(config: ResourceConfig) -> None:
    """Apply CPU thread limits.

    Uses 50% of available cores by default to keep the system responsive.
    """
    if config.cpu_threads > 0:
        num_threads = config.cpu_threads
    else:
        # Default: use ~50% of available CPU cores
        cpu_count = os.cpu_count() or 4
        num_threads = max(2, cpu_count // 2)

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(config.cpu_interop_threads)

    logger.debug(
        f"CPU limits: {num_threads} threads, "
        f"{config.cpu_interop_threads} interop threads "
        f"(total cores: {os.cpu_count()})"
    )


def _apply_gpu_limits(config: ResourceConfig) -> None:
    """Apply GPU memory limits.

    Sets max memory fraction and enables memory-efficient settings.
    """
    if not torch.cuda.is_available():
        logger.debug("No GPU available, skipping GPU limits")
        return

    # Set max memory fraction (prevents OOM by limiting allocation)
    try:
        torch.cuda.set_per_process_memory_fraction(
            config.gpu_max_memory_fraction, device=0
        )
        logger.debug(
            f"GPU memory fraction set to {config.gpu_max_memory_fraction:.0%}"
        )
    except Exception as e:
        logger.warning(f"Could not set GPU memory fraction: {e}")

    # Enable memory-efficient settings
    try:
        # Empty cache to start fresh
        torch.cuda.empty_cache()
    except Exception:
        pass
