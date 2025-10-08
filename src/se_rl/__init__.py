"""
SE-RL Framework: Self-Evolutional Reinforcement Learning
=======================================================

A comprehensive framework for LLM-powered reinforcement learning in financial trading.

Author: AI Research Engineer
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Research Engineer"

# Core imports
from .core.framework import SERLFramework, SERLConfig
from .core.performance_buffer import PerformanceBuffer
from .core.instruction_population import InstructionPopulation

# LLM components
from .llm.component_generator import LLMComponentGenerator, ComponentConfig
from .llm.integration import LLMIntegration, LLMConfig, LLMFactory
from .llm.prompts import PromptTemplates

# Data pipeline
from .data.pipeline import FinancialDataPipeline, DataConfig
from .data.feature_engineering import FeatureEngineer
from .data.normalization import DataNormalizer

# RL components
from .rl.trainer import RLTrainer, TrainingConfig
from .rl.agent import RLAgent
from .rl.metrics import FinancialMetrics

# Environments
from .environments.static_env import StaticEnvironment
from .environments.dynamic_env import DynamicEnvironment
from .environments.multi_agent_env import MultiAgentEnvironment

# Utils
from .utils.logger import setup_logging
from .utils.config_manager import ConfigManager

__all__ = [
    # Core
    'SERLFramework', 'SERLConfig', 'PerformanceBuffer', 'InstructionPopulation',
    # LLM
    'LLMComponentGenerator', 'ComponentConfig', 'LLMIntegration', 'LLMConfig', 'LLMFactory', 'PromptTemplates',
    # Data
    'FinancialDataPipeline', 'DataConfig', 'FeatureEngineer', 'DataNormalizer',
    # RL
    'RLTrainer', 'TrainingConfig', 'RLAgent', 'FinancialMetrics',
    # Environments
    'StaticEnvironment', 'DynamicEnvironment', 'MultiAgentEnvironment',
    # Utils
    'setup_logging', 'ConfigManager'
] 