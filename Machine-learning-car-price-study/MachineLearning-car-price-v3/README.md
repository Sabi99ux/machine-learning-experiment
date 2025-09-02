````markdown
# Car Price Prediction (Machine Learning Experiment) - Version 3

This project is the **third version** of a machine learning experiment to predict car prices based on multiple features such as brand, model, year, engine size, fuel type, transmission, mileage, and more.  
In this version, the main model is switched to **LightGBM Regressor**, which is faster and more efficient than XGBoost when handling large datasets.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/Sabi99ux/machine-learning-experiment.git
cd machine-learning-experiment/mechine-learning-car-price-study/MechineLearning-car-price-V3
````

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset used in this project contains up to **5 million rows** in the full version.
For demonstration purposes, only a **dummy dataset** is included in this repository for easier execution.

The full dataset can be accessed here:
👉 [Google Drive](https://drive.google.com/file/d/1i2SxpN4k-DN5RJ9P8whFiS4y0iuOlPRg/view?usp=drive_link)

**Raw dataset columns include:**

* brand, model, year, engine\_size, fuel\_type, transmission, mileage, body\_type, color, drive\_type, doors, seats, price, vehicle\_age, tax\_status

During preprocessing, categorical features are **encoded**, and additional **engineered features** are created such as `mileage_per_year`, `mileage_log`, `engine_per_seat`, `features_count`, `is_premium_brand`, `freq_brand`, and `age_x_mileage`.

---

## What's New in Version 3

* Model switched from **XGBoost (MultiOutputRegressor)** → **LightGBM Regressor**
* Focus on **single-target regression** (predicting car price only)
* Added **early stopping** to reduce overfitting
* Evaluation results are saved in **JSON format** for easier analysis
* Dummy dataset → enables quick demo, but not accurate compared to the full dataset
  ⚠️ **Disclaimer**: Predictions are not representative of real-world values

---

## Training the Model

Run:

```bash
python train.py
```

The script will train a **LightGBM Regressor** and save the model as a `.pkl` file.

---

## Evaluating the Model

Run:

```bash
python evaluate.py
```

The evaluation will generate metrics such as **RMSE** and **R² Score**, and save the results into a JSON file inside the `Result/` folder.

---

## Alternative: Jupyter Notebook

For users who prefer working with **Jupyter Notebook**, Version 3 also includes `train.ipynb` and `evaluate.ipynb`.
These notebooks are separated into multiple cells to make the workflow easier to understand and modify.

---

## Requirements

All dependencies are listed in `requirements.txt` with fixed versions to ensure compatibility.

---

## Notes

* The dataset included here is a **dummy dataset** for demonstration.
* The full dataset (\~5 million rows) is available on Google Drive.
* This project is intended for **educational and experimental purposes only**.
* Since the dataset is dummy, predictions may differ significantly from real-world values.
* Still learning—apologies if there are shortcomings 🙏

---

```
