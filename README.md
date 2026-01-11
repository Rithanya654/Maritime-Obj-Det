Maritime Object Detection with YOLOv8

A production-ready object detection system for maritime environments using YOLOv8-Large, trained on the SeaDronesSee dataset.
The model achieves 91.1% mAP@50 while maintaining real-time inference performance, enabling reliable detection of swimmers, boats, jetskis, lifesaving appliances, and buoys from aerial drone footage.

Table of Contents

Overview

Key Capabilities

Technical Highlights

Performance Metrics

Overall Performance

Per-Class Performance

Training Convergence

Dataset

Dataset Specifications

Data Processing Pipeline

Processed Dataset Statistics

Class Distribution

Installation

System Requirements

Setup Instructions

Required Dependencies

Verify Installation

Usage

Training

Inference

Model Export

Evaluation

Model Architecture

YOLOv8-Large Specifications

Feature Extraction Hierarchy

Training Pipeline

Data Processing Workflow

Training Configuration

Training Hardware Specifications

Learning Rate Schedule

Loss Functions

Experimental Results

Overview

This project was developed during an internship to build a state-of-the-art maritime surveillance system capable of detecting critical objects in complex ocean environments using YOLOv8-Large.

The system is optimized for real-world deployment, handling challenges such as:

Small object detection (swimmers, buoys)

High-resolution aerial imagery

Occlusions and boundary objects

Class imbalance in maritime datasets

Application Areas

Search and Rescue Operations
Rapid detection of swimmers and lifesaving equipment in emergency scenarios.

Maritime Surveillance
Automated monitoring of boats, jetskis, and marine traffic from UAVs.

Beach Safety Systems
Real-time tracking of swimmers and water activities to enhance public safety.

Marine Research & Analytics
Automated detection and counting of maritime objects for large-scale studies.

This repository provides a fully reproducible pipeline, covering dataset preprocessing, training, evaluation, inference, and deployment-ready exports.

Key Capabilities

High Accuracy

91.1% mAP@50

91.2% Precision

89.5% Recall

Real-Time Performance

~45 ms inference per 640×640 image on NVIDIA T4 GPU

Production-Ready Training

Optimized hyperparameters

Early stopping and checkpointing

Extensive validation and testing

Scalable Data Pipeline

COCO → YOLO conversion

High-resolution image tiling with overlap

Automated label validation and cleanup

Experimentally Verified

Multiple training versions

Resolution comparison studies

Detailed performance logs

Technical Highlights

Smart image tiling with overlap handling to avoid boundary detection loss

Robust label validation to eliminate corrupt annotations

Maritime-specific augmentation strategy (no vertical flips or perspective distortion)

Anchor-free detection with YOLOv8 decoupled head

Cosine annealing learning rate with warm-up

Extensive experimental comparisons (v1.0 – v3.0)
