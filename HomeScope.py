import json
import streamlit as st
import pandas as pd
import numpy as np
import dill
import gdown

st.title('HomeScope: California Median Price Forecast')


st.write('Test Message')


#with open('rfr.pkl', 'rb') as f:
#    reloaded_model = dill.load(f)


def get_model():
    
    url = 'https://drive.google.com/file/d/1Yfd5ZHSbxjCcq7er3z-pnmj6WvxISbKR/view?usp=sharing'
    output = 'HomeScope.pkl'
    gdown.download(url, output, quiet=False, fuzzy=True)
    with open('HomeScope.pkl', 'rb') as f:
        reloaded_model = dill.load(f)

    return reloaded_model


reloaded_model = get_model()




with open('rfr_info.json') as f:
    model_info = json.load(f)
    side_bar_options = model_info.get('options')
    options = {}
    for key, value in side_bar_options.items():
        if key in ['ocean_proximity']:
            options[key] = st.sidebar.selectbox(key, value)
        else:
            min_val, max_val = value
            current_value = (min_val + max_val)/2
            options[key] = st.sidebar.slider(key, min_val, max_val, value=current_value)


st.write(options)


if st.button('Predict'): 
  
    # Convert options to df 
    df = pd.Series(options).to_frame().T
    df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    st.write(df)
    y_hat = reloaded_model.predict(df)
    st.write(f'The predicted median house value is: ${y_hat[0]:,}')
