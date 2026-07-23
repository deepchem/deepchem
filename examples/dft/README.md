This directory contains examples demonstrating DeepChem's Density Functional Theory (DFT) capabilities with neural network exchange-correlation functionals.

## Overview

DeepChem implements neural network-based XC functionals that can be trained to replace or augment traditional density functionals. This approach, based on the paper ["Learning the exchange-correlation functional from nature with fully differentiable density functional theory"](https://arxiv.org/abs/2309.15985), enables learning XC functionals from high-quality quantum chemistry data.

## Available Examples

### 1. `neural_xc_tutorial.py`
Comprehensive tutorial demonstrating:
- **NNLDA**: Neural Local Density Approximation functional
- **NNPBE**: Neural PBE (GGA) functional  
- **HybridXC**: Combining traditional and neural functionals

**Key Concepts Covered:**
- Creating custom neural network architectures for XC functionals
- Understanding LDA vs GGA functional families
- Hybrid functional mixing strategies
- Training considerations for neural XC functionals

**Usage:**
```bash
python neural_xc_tutorial.py
Requirements:

PyTorch

DeepChem

DQC (Differentiable Quantum Chemistry)

Neural XC Functional Families
Local Density Approximation (LDA)
Depends only on electron density ρ(r)

Fast but less accurate

Good for prototyping and learning

DeepChem Implementation: NNLDA

Generalized Gradient Approximation (GGA)
Depends on density ρ(r) and gradient ∇ρ(r)

More accurate than LDA

Captures non-local effects

DeepChem Implementation: NNPBE

Hybrid Functionals
Mix traditional and neural functionals

Formula: E_xc = α × E_traditional + (1-α) × E_neural

Allows gradual transition to learned functionals

DeepChem Implementation: HybridXC

Current Limitations
Limited functional types: Only LDA and GGA neural implementations

No meta-GGA: Missing SCAN, M06-L, etc.

No range-separated hybrids: ωB97X, CAM-B3LYP not implemented

Scalability: Large systems may require optimization

Future Directions
Potential contributions (great for GSoC projects!):

Implement neural meta-GGA functionals (SCAN, r²SCAN, M06-L)

Add range-separated hybrid support (ωB97X, LC-ωPBE)

Improve GPU acceleration for large molecular systems

Create comprehensive benchmark suite

Add training utilities and pretrained models

Extend to periodic systems (solids, surfaces)

References
Kasim & Vinko, "Learning the exchange-correlation functional from nature with fully differentiable density functional theory", Physical Review Letters 127.12 (2021)

DeepChem DFT Infrastructure Paper: arxiv.org/abs/2309.15985

Contributing
Contributions are welcome! See DeepChem's contribution guidelines.

For questions about DFT examples, please post in the DeepChem Forum.

Author: Utkarsh Khajuria (@UtkarsHMer05)
Date: December 2025
