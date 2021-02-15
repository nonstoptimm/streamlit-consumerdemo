""" Consumer Insights Demo """
import pandas as pd
import streamlit as st
import json
import requests
import logging
from PIL import Image
from datetime import datetime
import glob
import os
import sys

# Import custom packages
sys.path.append('./')
import helper

# Logo
logo = Image.open('Microsoft-Logo.png')

### SIDE BAR
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.sidebar.image(logo, use_column_width=True, output_format='PNG')
st.sidebar.subheader("Consumer Insights Dashboard")
st.sidebar.markdown(
"""
Cluster your data and gain insights!

\n\n
""")

# Slider
pca_selection = st.sidebar.slider("Number of PCA", min_value=1, max_value=19, value=10)
cluster_selection = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=6)

# Credits
st.sidebar.subheader("Made by Timm Walz\nhttps://github.com/nonstoptimm\nMicrosoft Consulting Services")

### MAIN AREA
# Import and process data
df_exp_demographic, df_exp_clean, expenditures_scaled = helper.process_data()
demo_expander = st.beta_expander("Input consumer data")
demo_expander.write('In this section, the raw consumer data is displayed. The feature selection happened based on common demographic data and expenditure data.')
demo_expander.subheader('Demographic data')
demo_expander.dataframe(df_exp_demographic)
demo_expander.subheader('Expenditure data')
demo_expander.dataframe(df_exp_clean)

# Explained variance
var_expander = st.beta_expander("Explained variance")
var_expander.write('First, we create a plot to see how many principal components we need to picture the data in a way, that there are less dimensions, but still enough information left to gain valuable and realistic insights')
fig_explained_variance = helper.get_explained_variance_plot(expenditures_scaled)
var_expander.plotly_chart(fig_explained_variance)

# PCA
components, fig_pca = helper.get_pca(expenditures_scaled, df_exp_demographic, pca_selection)

# Cluster
df_pca_cluster, people_data, = helper.get_cluster(components, df_exp_demographic, df_exp_clean, cluster_selection)
cluster_summaries, scaled = helper.assemble_cluster_report(people_data)

# Cluster Summaries
summary_expander = st.beta_expander("Cluster summaries")
summary_expander.write('Below, you see summary statistics for every cluster given averaged demographic information and consumer behavior.')
summary_expander.subheader('Demographic information')
summary_expander.dataframe(cluster_summaries[helper.cols_dem].style.highlight_max(axis=0))
summary_expander.subheader('Expenditure information')
summary_expander.dataframe(cluster_summaries[helper.plot_clusters].style.highlight_max(axis=0))

# Spider plot
spider_expander = st.beta_expander("Spiderplot of cluster values", expanded=True)
spider_expander.write('Below, you see scatter polar plot, or also called "spiderplot" to illustrate the dimensions of each variable given a cluster. This helps us to better compare the consumer behavior in one place.')
spider_select = helper.plot_clusters
spider_options = spider_expander.multiselect(
    'Select attributes to be displayed',
    spider_select,
    spider_select)
cluster_options = spider_expander.multiselect(
    'Select clusters to be displayed',
    list(range(0, cluster_selection)),
    list(range(0, cluster_selection)))
    
fig_spider = helper.get_spiderplot(scaled.iloc[cluster_options], spider_options)
spider_expander.plotly_chart(fig_spider)