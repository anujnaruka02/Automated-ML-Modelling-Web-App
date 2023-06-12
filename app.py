import streamlit as st
import pandas as pd
import os
import json
import requests
from streamlit_lottie import st_lottie

import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
	st.image("https://media.istockphoto.com/id/844535726/photo/presentation-about-machine-learning-technology-scientist-touching-screen-artificial.jpg?s=612x612&w=0&k=20&c=zPVl1sUnQisKrcKgBaMUUl6yFr7ybOne9UPHDnIizL8=")
	st.title("AutoStreamML")
	choice = st.radio("Navigation",["Upload","Profiling","Machine Learning Analysis","Download"])
	st.info("A web-app to build an automated ML pipeline. Just feed the dataset and let the app do the magic!")



from pycaret.classification import setup, compare_models, pull, save_model

if os.path.exists("sourcedata.csv"):
	df = pd.read_csv("sourcedata.csv", index_col = None)

if choice == "Upload":
	st.title("Upload your data for modelling")
	file = st.file_uploader("Upload your Dataset Here")
	if file:
		df = pd.read_csv(file, index_col = None)
		df.to_csv("sourcedata.csv", index = None)
		st.dataframe(df)

if choice == "Profiling":
	st.title("Automated Exploratory Data Analysis")
	profile_report = df.profile_report()
	st_profile_report(profile_report)

if choice == "Machine Learning Analysis":
	st.title("Report of all machine learning models")
	target = st.selectbox("Select your target", df.columns)
	if st.button("Train model"):
		setup(df, target = target) #silent = true
		setup_df = pull()
		st.info("This is the ML Experiment settings")
		st.dataframe(setup_df)
		best_model = compare_models()
		compare_df = pull()
		st.info("This is the ML Model")
		st.dataframe(compare_df)
		best_model
		save_model(best_model, 'best_model')

if choice == "Download":
	with open("best_model.pkl", 'rb') as f:
		st.download_button("Download the Model", f, "trained_model.pkl")

# animation
def load_lottieurl(url: str):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_u8jppxsl.json")

st_lottie(lottie_hello, key = "Hi")