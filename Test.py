import streamlit as st
from pycaret.classification import load_model

pipeline = load_model("Location of the trained model to be added here")
pipeline
