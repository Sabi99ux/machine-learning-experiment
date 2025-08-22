---

# Car Price Prediction (Machine Learning Experiment) - Version 2

This project is the second version of a machine learning experiment to predict car prices and other car-related attributes based on multiple features such as brand, model, year, engine size, fuel type, transmission, mileage, and more.
It uses **XGBoost with MultiOutputRegressor** as the main prediction model.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/Sabi99ux/machine-learning-experiment.git
cd machine-learning-experiment/mechine-learning-car-price-study/MechineLearning-car-price-V2
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset used in this project contains millions of rows in the full version.
For demonstration purposes, only a **dummy dataset** is included in this repository.

The full dataset can be accessed via this link: [Google Drive](https://drive.google.com/file/d/1i2SxpN4k-DN5RJ9P8whFiS4y0iuOlPRg/view?usp=drive_link)

**Columns include (raw dataset):**

* brand
* model
* year
* engine\_size
* fuel\_type
* transmission
* mileage
* body\_type
* color
* drive\_type
* doors
* seats
* price
* vehicle\_age
* tax\_status

During training, categorical features are automatically **encoded** and several **engineered features** are generated to improve model performance.

---

## New Features in Version 2

This version introduces several important improvements:

* Uses **XGBoost Regressor** with **MultiOutputRegressor** as the main model.
* Supports **multi-target prediction** (not only predicting price but also other attributes like engine size, mileage, vehicle age, etc).
* Dataset scaled up to **\~5 million rows** in the full version (dummy dataset included here).
* Added multiple **engineered features** for better prediction accuracy:

  * `mileage_per_year` → ratio of mileage per year.
  * `mileage_log` → logarithmic transformation of mileage.
  * `engine_per_seat` → engine capacity per seat.
  * `features_count` → number of additional features (GPS, sunroof, etc).
  * `is_premium_brand` → indicator for premium brands.
  * `freq_brand` → frequency of brand occurrence.
  * `age_x_mileage` → interaction between vehicle age and mileage.

---

## Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Load and preprocess the dataset
* Encode categorical features automatically
* Generate engineered features
* Train an **XGBoost Regressor** wrapped with **MultiOutputRegressor**
* Save the trained model and encoders as `.pkl` files

---

## Evaluating the Model

Run the evaluation script:

```bash
python evaluate.py
```

This will:

* Load the trained model and encoders
* Test predictions on the dataset
* Display performance metrics such as **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R² Score** (per target and overall)

---

## Requirements

All dependencies are listed in `requirements.txt` with fixed versions to ensure compatibility.

---

## Notes

* The dataset in this repository is only a dummy dataset for demonstration.
* The full dataset with \~5 million rows can be accessed via Google Drive.
* This project is still experimental and intended for **educational purposes only**.
* Apologies if there are still shortcomings, as I am also still learning. 🙏


---
