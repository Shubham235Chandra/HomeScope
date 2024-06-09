
# HomeScope: California Median House Price Prediction

## Overview

HomeScope is a data science project focused on predicting median house prices in California using a Random Forest Regressor model. It incorporates a variety of data preprocessing techniques, machine learning models, and deployment strategies to provide an intuitive interface for house price prediction.

## Project Structure

- `housing.csv`: Dataset used for training and testing the model.
- `Link.docx`: Document containing a link to the deployed Streamlit app.
- `part1.ipynb`: Jupyter notebook for initial analysis and preprocessing.
- `preprocessing.ipynb`: Jupyter notebook dedicated to data preprocessing.
- `requirements.txt`: Specifies Python dependencies required for the project.
- `rfr_info.json`: JSON file with details on the Random Forest Regressor model and input features.
- `cal_predict.py`: Python script for Streamlit app deployment.
- `deploy.ipynb`: Jupyter notebook outlining deployment steps.
- `HomeScope.py`: Main script for the Streamlit app.

## Setup

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HomeScope.git
   cd HomeScope
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To start the Streamlit app, run:
```bash
streamlit run HomeScope.py
```
The application will be accessible at `http://localhost:8501`.

## Usage

1. Navigate to the deployed app or launch the app locally.
2. Adjust the input parameters using the sidebar options.
3. Click the "Predict" button to receive the predicted median house price.

## Model Information

The project uses a Random Forest Regressor. The `rfr_info.json` file contains detailed information about the model, including input features and their respective ranges.

### Input Features

- `longitude`: Longitude of the location.
- `latitude`: Latitude of the location.
- `housing_median_age`: Median age of the houses.
- `total_rooms`: Total number of rooms in the houses.
- `total_bedrooms`: Total number of bedrooms in the houses.
- `population`: Population in the area.
- `households`: Number of households.
- `median_income`: Median income of the residents.
- `ocean_proximity`: Proximity to the ocean.

## Contributing

Contributions are welcome! Please read the contributing guidelines first.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Dataset: California Housing Prices dataset.
- Streamlit for providing a platform to deploy data science apps.
- Scikit-learn for machine learning algorithms.

## Links

- [Deployed App](URL_to_deployed_app)
- [GitHub Repository](https://github.com/yourusername/HomeScope)
