import streamlit as st
import pickle as pk

model = pk.load(open('model.pkl','rb'))  # This is a pipeline

review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    result = model.predict([review])  # Pass raw text
    if result[0] == 0:
        st.write('Negative Review')
    else:
        st.write('Positive Review')
    proba = model.predict_proba([review])
    st.write(f"Confidence: {proba}")
