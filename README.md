# Music Instrument Classification

This repository contains the code and documentation for a project on **automatic multilabel musical instrument classification** using deep learning techniques. The project was developed within the course *Selected Topics in Music and Acoustic Engineering* and is based on the **MedleyDB** dataset.

## Overview

The system is designed to detect and classify multiple instruments active in audio segments. The pipeline includes:

- **Dataset parsing and preprocessing** from multitrack recordings (MIX, STEM, RAW files)
- **Label grouping** into 17 macro-categories for better generalization
- **Audio segmentation** using pitch annotations or energy-based heuristics
- **Mel-spectrogram extraction** as input representation
- **Convolutional Neural Network (CNN)** training with multilabel output
- **Model evaluation** with classification metrics and confusion analysis
- **Metadata exploration** to analyze genre-instrument associations and co-occurrence

## Features

- Flexible data loading and configuration
- Iterative stratified data splitting
- Label binarization for multilabel classification
- Data augmentation for class balancing
- Robust evaluation including per-class precision, recall, F1-score
- Co-occurrence analysis for instruments and genres

## Dataset

The project uses the [MedleyDB](https://medleydb.weebly.com/) dataset (version 1.0), which includes multitrack recordings with pitch annotations and stem-level metadata.

> The dataset is not included. Please download it manually from the official source and configure the correct path in the code.

## Usage

The workflow is implemented in a single Python notebook and includes:

1. Metadata parsing and label assignment
2. Audio extraction and mel-spectrogram computation
3. Train/validation/test splitting (60/24/16)
4. CNN training with early stopping and checkpoints
5. Performance evaluation on single and multilabel samples
6. Genre and instrument co-occurrence analysis

## Output

The trained model achieves high performance on balanced macro-categories, with strong generalization on real polyphonic music. Classification reports, confusion matrices, and co-occurrence heatmaps are available in the final report.

## Authors

Andrea Crisafulli, Giacomo De Toni, Marco Porcella, Gianluigi Vecchini  
Politecnico di Milano, 2025  
Advisor: Prof. Julio Carabias
