# A Hybrid Multi-Factor Network with Dynamic Sequence Modeling for Early Warning of Intraoperative Hypotension

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)  

This repository provides the implementation of **HMF**, a Hybrid Multi-Factor framework for dynamic intraoperative hypotension (IOH) prediction. 

---

## 🔍 Overview

The **HMF framework** is designed to predict IOH using Mean Arterial Pressure (MAP) as a key indicator. Traditional approaches often use static models that fail to capture the dynamic nature of physiological signals. In contrast, our framework employs advanced techniques to address challenges such as distributional drift and temporal dependencies in physiological data.

### Key Features
- **Transformer Encoder:** Captures the temporal evolution of MAP through patch-based input representation.
- **Symmetric Normalization and De-Normalization:** Mitigates distributional drift in data to ensure robust performance across varying conditions.
- **Sequence Decomposition:** Disaggregates input signals into trend and seasonal components for improved sequence modeling.
- **State-of-the-Art Performance:** Extensively validated on two real-world datasets, outperforming competitive baselines.

---

## 📥 Installation

### Prerequisites

- Python 3.7 or later
- PyTorch >= 1.10
- Other dependencies listed in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HMF-IOH.git
   cd HMF-IOH
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Data Preparation

- Place your physiological signal datasets in the `data/` directory.
- Ensure the data format follows the structure outlined in `docs/data_format.md`.

### Training

To train the model, run:
```bash
python train.py --config configs/default.yaml
```

### Evaluation

Evaluate the trained model using:
```bash
python evaluate.py --model checkpoints/best_model.pth --data data/test_data.csv
```

### Configuration

Modify `configs/default.yaml` to customize training parameters, such as learning rate, batch size, and model architecture.

---

## 📊 Results

The HMF framework demonstrates state-of-the-art performance on two real-world datasets. See the [paper](https://arxiv.org/abs/2409.11064) for detailed experimental results.

---

## 🤝 Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file before submitting pull requests.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---


```
