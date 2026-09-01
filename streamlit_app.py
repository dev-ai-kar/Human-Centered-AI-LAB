import streamlit as st

Lab1 = st.Page('Lab1.py', title='Lab1', icon=':material/add_circle:')
Lab2 = st.Page('Lab2.py', title='Lab2', icon=':material/add_circle:')

pg = st.navigation([Lab1, Lab2])
st.set_page_config(page_title='HCAI Labs', page_icon=':material/edit:')
pg.run()