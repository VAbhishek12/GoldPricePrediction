# Gold Price Prediction

## Overview
This project aims to forecast gold prices using machine learning techniques. By leveraging historical gold price data, various regression models are trained and evaluated to predict future price movements. The core of this project is implemented in a Jupyter Notebook, demonstrating the data preprocessing, model training, and evaluation phases.

## Features
- **Data Collection & Preprocessing:** Efficient handling and cleaning of historical gold price data using `pandas`.
- **Exploratory Data Analysis (EDA):** Visualizations using `matplotlib` and `seaborn` to understand trends, patterns, and correlations within the dataset.
- **Model Training:** Implementation of regression models (e.g., Random Forest Regressor) from `scikit-learn` to predict gold prices.
- **Model Evaluation:** Assessment of model performance using key metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Deployment Script:** A Python script (`Deployment.py`) for potentially deploying the trained model.

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib (for model serialization/deserialization)

## Files in this Repository
- `gold_price_forecast.ipynb`: The main Jupyter Notebook containing the data analysis, model training, and evaluation code.
- `Deployment.py`: A Python script for deploying the trained model (e.g., as an API endpoint or for batch predictions).
- `gold_forecast_joblib`: Likely a saved machine learning model (e.g., a `.joblib` file) after training.
- `README.md`: This file.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/VAbhishek12/GoldPricePrediction.git](https://github.com/VAbhishek12/GoldPricePrediction.git)
    cd GoldPricePrediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you don't have a `requirements.txt` file, you can create one by running `pip freeze > requirements.txt` after installing all dependencies listed in the 'Technologies Used' section within your virtual environment.*

## Usage

1.  **Run the Jupyter Notebook:**
    To explore the data analysis, model training, and evaluation, open the `gold_price_forecast.ipynb` notebook:
    ```bash
    jupyter notebook
    ```
    Follow the steps within the notebook to understand the project flow.

2.  **Using the Deployment Script:**
    The `Deployment.py` script can be used to load the trained model and make predictions.
    ```bash
    python Deployment.py
    ```
    *(Further instructions on how to use `Deployment.py` would depend on its specific implementation, e.g., if it takes command-line arguments or runs a web server.)*

## Results

### Gold Price Trend
![Gold Price Trend](images/gold_price_trend.png)

### Model Prediction vs. Actual Prices
![Prediction vs Actual](images/prediction_vs_actual.png)

### Error Distribution
![Error Distribution](images/error_distribution.png)
