# Regression Model for Skin Temperature Prediction

This project implements a regression model to predict skin temperature (Tskin) based on insulation thickness and heat flux using a Random Forest Regressor.

## Files

- `aiproject.py`: Main Python script containing the regression model implementation.
- `clean.csv`: Cleaned dataset used for training and testing.
- `clean.py`: Script for data cleaning (if applicable).
- `ansys_results.csv`: Original ANSYS simulation results.
- `ansys_results_ourmodel.csv`: Results from our model.
- `design_space_contour.png`: Visualization of the design space contour plot.

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, scikit-learn

Install dependencies with:
```
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run the main script:
```
python aiproject.py
```

This will load the data, train the model, evaluate performance, and generate visualizations.

## Model Performance

- RMSE: [Insert RMSE value]
- R²: [Insert R² value]

## License

[Add license if applicable]