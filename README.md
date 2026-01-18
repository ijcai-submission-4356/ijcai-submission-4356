# Large Language Model (LLM) as an Excellent Reinforcement Learning Researcher in Financial Single-Agent and Multi-Agent Scenarios

A comprehensive implementation of the Self-Evolutional Reinforcement Learning (SE-RL) framework for automated RL algorithm design and optimization in financial order execution, as described in the research paper.

Due to double-blind review reasons and folder size limitations, we will open source our training weights after the review is complete.

## 🚀 Overview

The SE-RL framework addresses two fundamental challenges in RL-based financial trading:
1. **Slow research pace** - Traditional RL methods evolve slowly compared to other AI domains
2. **Idealistic market assumptions** - Existing methods ignore the impact of orders on market dynamics

The framework uses Large Language Models (LLMs) to automatically design, train, and iteratively optimize RL algorithms through a dual-loop architecture:

- **Outer Loop**: LLM research capability evolution
- **Inner Loop**: Execution agent training in hybrid environments

## 🏗️ Architecture

### Core Components

1. **SE-RL Framework** (`se_rl_framework.py`)
   - Main framework orchestrating the entire process
   - Implements Algorithm 1 from the paper
   - Manages outer and inner loops

2. **LLM Component Generator** (`llm_generator.py`)
   - Generates reward functions, network architectures, and imagination modules
   - Uses sophisticated prompting techniques
   - Includes code validation and retry mechanisms

3. **Financial Data Pipeline** (`financial_data_pipeline.py`)
   - Downloads and processes CSI100 and NASDAQ100 data
   - Engineers comprehensive financial features
   - Creates PyTorch datasets for training

4. **RL Training System** (`rl_trainer.py`)
   - Implements RL agents and training environments
   - Calculates financial performance metrics (PA, WR, GLR, AFI)
   - Supports both static and dynamic environments

5. **Main Execution Script** (`main.py`)
   - Ties all components together
   - Provides command-line interface
   - Generates comprehensive reports

### Key Features

- **Dual-Level Enhancement Kit (DEK)**: High-level prompt refinement and low-level weight optimization
- **Hybrid Environment Training**: Combines static historical data with dynamic market simulation
- **Multi-Agent Market Simulation**: Realistic modeling of market impact and agent interactions
- **Sophisticated Prompting**: Few-shot learning, chain-of-thought, and advanced prompt engineering
- **Financial Metrics**: Comprehensive evaluation using industry-standard metrics
