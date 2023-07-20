import streamlit as st
from pycaret.classification import load_model

pipeline = load_model("C:/Users/anujn/OneDrive/Desktop/trained_model")
pipeline