# Titanic Survival Prediction

This repository contains a Python script for predicting Titanic survival outcomes using the Titanic dataset. The project uses various data preprocessing techniques and machine learning algorithms to train a model that can predict whether a passenger survived or not.

## Overview

The script processes the Titanic dataset by handling missing values, encoding categorical features, and training an XGBoost classifier. The model is then used to make predictions on the test dataset.

## Features

- Data loading and preprocessing (e.g., handling missing values, label encoding)
- Model training using XGBoost classifier
- Prediction on the test dataset
- Results saved to a CSV file (`resultfile.csv`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the `train.csv` and `test.csv` files in the same directory as the `titanic.py` script.
2. Run the script:
   ```bash
   python titanic.py
   ```
3. The script will output the predictions to a file named `resultfile.csv`.

## Dependencies

- pandas
- numpy
- matplotlib
- xgboost
- scikit-learn
- tabulate

Make sure to install these packages using pip:
```bash
pip install pandas numpy matplotlib xgboost scikit-learn tabulate
```

## Results

The model predictions are saved to `resultfile.csv`, which contains the survival predictions for each passenger in the test dataset.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) - The dataset used for this project.
