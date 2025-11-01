# Stable Diffusion Implementation - Learning Project

## Overview

This repository contains my implementation of Stable Diffusion in PyTorch, created as a learning exercise by following along with educational materials.

**Learning Source:** 
- Video: ["Coding Stable Diffusion from scratch in PyTorch"](https://www.youtube.com/watch?v=ZBKpAp_6TGI) by Umar Jamil (5-hour course)
- Original Code: [hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion)

## Learning Objectives

- Understand the architecture of diffusion models
- Implement key components: VAE, UNet, and diffusion process
- Gain hands-on experience with PyTorch
- Learn image generation techniques

## Components Implemented

1. **VAE (Variational Autoencoder)**
   - Encoder: Compresses images into latent space
   - Decoder: Reconstructs images from latent representations

2. **UNet**
   - Core denoising network
   - Residual blocks and attention mechanisms

3. **Attention**
   - Self-attention and cross-attention layers
   - Enables text-conditional generation

4. **DDPM (Denoising Diffusion Probabilistic Models)**
   - Noise scheduler
   - Sampling algorithms

5. **Diffusion Process**
   - Forward process: noise addition
   - Reverse process: iterative denoising

## Sample Output

Successfully generated images from text prompts:
- "A dog stretching on the floor, highly detailed" → Beagle-style image

## Development Notes

**Implementation time:** ~5 hours of coding and debugging

**Challenges:**
- Debugging tensor dimension mismatches
- Fixing mathematical formula implementations
- Correcting variable name typos
- Extensive AI-assisted debugging

## Acknowledgments

- [Umar Jamil](https://github.com/hkproj) for the excellent tutorial
- AI debugging tools for troubleshooting assistance
