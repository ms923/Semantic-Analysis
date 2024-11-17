
# Semantic Textual Similarity with Supervised and Unsupervised Learning

**Authors**: Tejansh Sachdeva and Mitaali Singhal

## Overview

Semantic Textual Similarity (STS) assesses the relationship between text pairs, playing a critical role in Natural Language Processing (NLP) applications like information retrieval, question answering, and summarization.

This repository provides the codebase and resources for our research project, which introduces a hybrid approach combining **Supervised Learning**, **Unsupervised Learning**, and **Ensembled Models** to achieve state-of-the-art performance in STS tasks.

Key highlights include:

- Use of traditional models like **SVR**, **LightGBM**, and **XGBoost**.
- Integration of **feedforward neural networks** for unsupervised learning.
- Novel ensemble models (**XGB + LightGBM**, **SVR + Neural Network**) for robust similarity prediction.
- Experiments conducted on benchmark datasets such as **SemEval 2012 Task 6**.

## Repository Structure

```plaintext
├── data/  
│   └── dataset.txt      # Primary dataset for experiments.  
├── supervised/  
│   ├── baseline/        # Baseline supervised models.  
│   ├── lightgbm/        # LightGBM implementation.  
│   ├── svr/             # Support Vector Regressor.  
│   └── xgboost/         # XGBoost implementation.  
├── unsupervised/  
│   └── feedfwd_nn/      # Feedforward neural network for unsupervised learning.  
├── ensembled/  
│   ├── xgb_lgm/         # Ensemble of XGBoost + LightGBM.  
│   └── svr_nn/          # Ensemble of SVR + Neural Network.  
├── requirements.txt     # Dependencies for the project.  
└── README.md            # Project documentation.  
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ms923/Semantic-Analysis.git
   cd Semantic-Analysis
   ```
2. Create a virtual environment (Python 3.11 recommended):

   ```bash
   python3.11 -m venv env  
   source env/bin/activate  # For Linux/Mac  
   env\Scripts\activate     # For Windows  
   ```
3. Install required dependencies:

   ```bash
   pip install -r requirements.txt  
   ```

## Usage

- **Supervised Models**: Run individual models (`svr`, `lightgbm`, `xgboost`) from the `supervised` folder.
- **Unsupervised Models**: Train the feedforward neural network in the `unsupervised` folder.
- **Ensembled Models**: Combine the strengths of different methods by running scripts in the `ensembled` folder.

## Results

- Achieved **state-of-the-art** correlations with human similarity judgments using Pearson and Spearman metrics.
- Detailed evaluation available in the paper accompanying this repository.

## Datasets

The `dataset.txt` file in the `data` folder contains the primary data used for training and evaluation.

## Citation

If you use this code or approach in your research, please cite:

```plaintext
@article{Tejansh2024STS,  
  title={Semantic Textual Similarity with Supervised and Unsupervised Learning: Applications of SVR and Ensembling},  
  author={Tejansh Sachdeva and Mitaali Singhal},  
  year={2024}  
}  
```

## License

This project is licensed under the MIT License.

---
