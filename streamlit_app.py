import streamlit as st

Lab1 = st.Page('Lab1.py', title='Lab1', icon=':material/add_circle:')
Lab2 = st.Page('Lab2.py', title='Lab2', icon=':material/add_circle:')
Lab3 = st.Page('Lab3.py', title='Lab3', icon=':material/add_circle:')
Lab4 = st.Page('Lab4.py', title='Lab4', icon=':material/add_circle:')
Lab5 = st.Page('Lab5.py', title='Lab5', icon=':material/cloud:')

pg = st.navigation([Lab1, Lab2, Lab3, Lab4, Lab5])
st.set_page_config(page_title='HCAI Labs', page_icon=':material/edit:')
pg.run()