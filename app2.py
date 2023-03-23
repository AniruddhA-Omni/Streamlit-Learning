import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl

# ML Model

# df = pd.read_csv(r"data/penguins_cleaned.csv")
# target = 'species'
# encode = ['sex', 'island']

# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]

# target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

# def target_encode(val):
#     return target_mapper[val]

# df['species'] = df['species'].apply(target_encode)

# X = df.drop(columns=['species'])
# y = df['species']

# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()
# model.fit(X, y)

# pkl.dump(model, open('models/penguins_model.pkl', 'wb'))



# Streamlit app

model = pkl.load(open('models/penguins_model.pkl', 'rb'))

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
""")

st.sidebar.header('User Input Features')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
penguins_raw = pd.read_csv(r"data/penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)


encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader('User Input parameters')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

pred = model.predict(df)
pred_proba = model.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[pred])

st.subheader('Prediction Probability')
st.write(pred_proba)