import streamlit as st
import pandas as pd
import numpy as np
import json
import dill
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

def download_file(url, output):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info(f"File downloaded successfully: {output}")
        return True
    except Exception as e:
        st.error(f"Error downloading the file: {e}")
        logging.error(f"Error downloading the file: {e}")
        return False

def is_valid_dill_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            dill.load(f)
        return True
    except Exception as e:
        logging.error(f"Invalid dill file: {e}")
        return False

def get_model():
    url = 'https://drive.google.com/uc?id=1Yfd5ZHSbxjCcq7er3z-pnmj6WvxISbKR'
    output = 'HomeScope.pkl'
    if not os.path.exists(output) or not is_valid_dill_file(output):
        if download_file(url, output):
            if not is_valid_dill_file(output):
                st.error("Downloaded file is not a valid dill file.")
                logging.error("Downloaded file is not a valid dill file.")
                return None
        else:
            return None
    try:
        with open(output, 'rb') as f:
            reloaded_model = dill.load(f)
        return reloaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        logging.error(f"Error loading the model: {e}")
        return None

reloaded_model = get_model()

if reloaded_model:
    st.title('HomeScope: California Median Price Forecast')

    try:
        with open('rfr_info.json') as f:
            model_info = json.load(f)
            side_bar_options = model_info.get('options')
            options = {}
            for key, value in side_bar_options.items():
                if key in ['ocean_proximity', 'income_cat']:
                    options[key] = st.sidebar.selectbox(key, value)
                else:
                    min_val, max_val = value
                    current_value = (min_val + max_val) / 2
                    options[key] = st.sidebar.slider(key, min_val, max_val, value=current_value)

        st.write(options)

        if st.button('Predict'):
            # Convert options to df
            df = pd.Series(options).to_frame().T
            df["income_cat"] = pd.cut(df["median_income"],
                                      bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                      labels=[1, 2, 3, 4, 5])

            y_hat = reloaded_model.predict(df)
            st.write(df)
            st.write(y_hat)
            # st.write(f'The predicted median house value is: ${y_hat[0]:,}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
else:
    st.error("Model could not be loaded.")
    logging.error("Model could not be loaded.")
