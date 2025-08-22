# Car Price Prediction (Machine Learning Experiment)

This project is a machine learning experiment to predict car prices based on various features such as brand, model, year, engine size, fuel type, transmission, mileage, and more.  
It uses **Linear Regression** as the prediction model.

## Installation

Clone this repository:

```bash
git clone https://github.com/Sabi99ux/machine-learning-experiment.git
cd machine-learning-experiment/mechine-learning-car-price-study/MechineLearning-car-price-V1
````

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is located in `Dataset/Dataset_car.csv` and contains a small sample of car information for demonstration purposes.

**Columns include:**

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

## Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Load the dataset
* Encode categorical features
* Train a **Linear Regression** model
* Save the trained model and encoders as `.pkl` files

## Evaluating the Model

Run the evaluation script:

```bash
python evaluate.py
```

This will:

* Load the trained model and encoders
* Test predictions on the dataset
* Show performance metrics such as **Mean Squared Error (MSE)** and **R² score**

## Requirements

All dependencies are listed in `requirements.txt` with specific versions to ensure compatibility.

## Notes

* The dataset here is a reduced version for demonstration.
* In real applications, you should train the model with a larger dataset for better accuracy.
* This project is for **educational purposes only**.
```
