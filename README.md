# ai4good-CNN_introduction

Overview
Over 80% of smallholder farms in Sub-Saharan Africa depend on cassava as a primary food source. Viral diseases are silently devastating these crops, and with limited access to agricultural experts, farmers have no fast, affordable way to diagnose what's wrong.
This project applies transfer learning and computer vision to classify cassava leaf images into one of five categories: four disease types or healthy. Built on 21,367 real field images sourced from Ugandan farms, the goal is a model accurate enough to run on a mobile camera — putting disease detection directly in a farmer's hands.

## The Problem

Cassava is the second-largest carbohydrate source in Sub-Saharan Africa
Viral diseases cause significant yield loss, threatening food security for millions
Expert-led visual inspection is labor-intensive, costly, and unscalable
Farmers capturing images on basic mobile devices need a lightweight, reliable classifier


## Approach
Four models were developed and compared iteratively, each building on the weaknesses of the last.
### Model 1: Baseline (Evaluation Only)

Architecture: ShuffleNet V2 x0.5 (pretrained, ImageNet weights)
No fine-tuning; evaluated directly on the validation set
Establishes a performance floor for comparison

### Model 2: Pre-built Fine-tuned

Architecture: ShuffleNet V2 x0.5
Fine-tuned with Adam optimizer (LR: 0.001), CrossEntropyLoss, 10 epochs
Establishes the gain from training vs. zero-shot transfer

### Model 3: ResNet-18

Architecture: ResNet-18 (deeper, more expressive than ShuffleNet)
Lower LR (0.0001) to preserve pretrained feature representations
Separate train/val transforms introduced: random crop and horizontal flip for augmentation
10 epochs

### Model 4: ResNet-50 (Best Model)

Architecture: ResNet-50
Richer augmentation: random crop, horizontal flip, vertical flip, ColorJitter
LR scheduler: StepLR (halves LR every 5 epochs for refined late-stage convergence)
Weighted loss to address class imbalance — minority disease classes weighted higher
15 epochs


## Key Design Decisions
DecisionRationaleShuffleNet- ResNetResNet's skip connections enable deeper feature learning; better suited for subtle inter-class differencesLR 0.001 → 0.0001Smaller steps preserve pretrained ImageNet weights while fine-tuning to cassavaAdded augmentationReal field images vary in angle, lighting, and crop- augmentation improves robustnessWeighted lossDataset is imbalanced; without weighting, the model ignores minority disease classesStepLR schedulerReduces LR mid-training to avoid overshooting in later epochs

## Tech Stack

PyTorch: model training and evaluation
TorchVision: pretrained architectures (ShuffleNet, ResNet-18, ResNet-50)
TorchMetrics: accuracy tracking and confusion matrix
Google Colab: GPU-accelerated training environment
Kaggle Dataset: 21,367 labeled cassava leaf images


## Skills Demonstrated

Transfer learning with pretrained CNNs
Iterative model development and ablation reasoning
Handling real-world class imbalance
Data augmentation pipeline design
Training loop implementation from scratch (forward pass, backprop, optimizer step)
Evaluation with confusion matrices


## Dataset
Cassava Leaf Disease Classification — Kaggle
21,367 labeled images across 5 classes, captured by Ugandan farmers using mobile devices under natural field conditions.
Built as part of a machine learning course project. Dataset sourced from a real Kaggle competition in partnership with Makerere University, Uganda.
