import configparser
import os
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cut_tree
import plotly.graph_objects as go
import multiprocessing

consum_features = ['alcohol', 'books', 'clothes',
	'elect',
	'food',
	'foodhome',
	'foodout',
	'foodwork',
	'gambling',
	'gas',
	'gasoline',
	'hlthbeau',
	'homefuel',
	'homeval2',
	'housuppl',
	'jewelry',
	'tailors',
	'telephon',
	'tobacco',
	'utility',
	'water']
    
cols_dem = ['age', 'hhsize', 'num_child', 'blsurbn', 'income', 'educatio', 'race', 'sex', 'work', 'marital']

median_cols = ['age']
skip = ['emptype', 'empstat', 'occup', 'age_group']
most_common = ['educatio', 'marital', 'empstat', 'occup', 'emptype', 'race']

plot_clusters = [
     'alcohol',
     'books',
     'clothes',
     'elect',
     'food',
     'foodhome',
     'foodout',
     'gas',
     'gasoline',
     'hlthbeau',
     'homeval2',
     'jewelry',
     'telephon',
     'tobacco',
     'utility',
     'water'
]

@st.cache
def remove_outliers(df_exp_clean):
    '''Remove outliers based on percentiles'''
    for column in list(df_exp_clean):
        percentiles = df_exp_clean[column].quantile([0.05, 0.95]).values
        df_exp_clean[column][df_exp_clean[column] <= percentiles[0]] = percentiles[0]
        df_exp_clean[column][df_exp_clean[column] >= percentiles[1]] = percentiles[1]
    return df_exp_clean

# Import consumer expenditure data
@st.cache
def process_data(data='Data_consumer_expenditure_survey.csv', data2='Data_Supplementary_price.csv'):
    # Expenditure Data
    df_exp = pd.read_csv(data).set_index('newid')
    df_exp['work'] = df_exp['nonwork'].replace('\\N', 0).astype(int)
    
    # Supplement data
    df_sup = pd.read_csv(data2)

    # Extract demographic information and assign it to a separate data frame
    df_exp_demographic = df_exp[cols_dem]
    
    # Extract the relevant information from the expenditure data frame
    df_exp_values = df_exp[consum_features]

    # Variables housuppl and toiletry seem to be missing or there are no values, so we remove them from the df
    df_exp_values.drop(['housuppl', 'foodwork', 'gambling', 'homefuel'], axis=1, inplace=True)

    df_exp_clean = df_exp_values.copy()

    # Run the function
    df_exp_clean = remove_outliers(df_exp_clean)

    # Standardizing the features to have less variance in the data and make the values across the variables more homogeneous
    expenditures_scaled = StandardScaler().fit_transform(df_exp_clean.values)

    return df_exp_demographic, df_exp_clean, expenditures_scaled, df_sup

def get_timeplot(df_sup):
    # Plot the supplement dataset to gain an overview on how prices develop over the past couple of years
    df_sup_grouped = df_sup.groupby(['year']).mean().reset_index()
    dfx = df_sup_grouped.drop(['quarter', 'date'], axis=1).melt('year', var_name='expenditures', value_name='price over year')
    g = sns.relplot(x="year", y="price over year", kind="line", hue='expenditures', data=dfx, height=6, aspect=2)
    return g

@st.cache
# Next, we create the principal components given an amount n and create a plot
# how the data looks like in dependence of the age
def get_pca(expenditures_scaled, df_exp_demographic, n_components):
    '''Get PCA and variance plot for specific n'''
    # Run PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(expenditures_scaled)
    total_var = pca.explained_variance_ratio_.sum() * 100
    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = 'Age'
    
    # Plot
    fig = px.scatter_matrix(
        components,
        color=df_exp_demographic['age'],
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.update_traces(diagonal_visible=False)
    return components, fig

@st.cache
def get_cluster(components, df_exp_demographic, df_exp_clean, n_clusters):
    merged_data = linkage(pd.DataFrame(components), method='ward')
    
    # Transform data set and attach principal components along with a cluster number
    cluster_cut = pd.Series(cut_tree(merged_data, n_clusters = n_clusters).reshape(-1,))
    df_pca_cluster = pd.concat([pd.DataFrame(components), cluster_cut], axis=1)
    df_pca_cluster.columns = [f'PC{index}' for index in range(1, len(components[0])+1)] + ["cluster_nr"]
    
    # Merge demographic information to principal components and the cluster number
    df_pca_cluster.index = df_exp_demographic.index
    people_data = df_exp_demographic.join(df_pca_cluster)
    people_data = people_data.join(df_exp_clean)
    people_data.sample(5)
    return df_pca_cluster, people_data

@st.cache
def assemble_cluster_report(people_data):
    '''Create cluster report based on principal components and average values for every variable'''
    cluster_summaries = pd.DataFrame()
    for column in cols_dem + list(people_data):
        if column in skip:
            continue
        elif column in median_cols:
            cluster_summaries[column] = people_data.groupby(["cluster_nr"])[column].median()
        elif column in most_common:
            cluster_summaries[column] = people_data.groupby(["cluster_nr"])[column].agg(pd.Series.mode)
        else:
            cluster_summaries[column] = people_data.groupby(["cluster_nr"])[column].mean()
    
    # Scale values to have a common scale (e.g. food is way higher than other values in comparison)
    scaled = (cluster_summaries[plot_clusters]-cluster_summaries[plot_clusters].min())/(cluster_summaries[plot_clusters].max()-cluster_summaries[plot_clusters].min())

    return cluster_summaries, scaled

@st.cache
def get_spiderplot(scaled, plot_clusters):
    '''Create spiderplot to illustrate expense dimensions'''
    fig = go.Figure()
    for index, row in scaled.iterrows():
        fig.add_trace(go.Scatterpolar(
              r=row.to_list(),
              theta=plot_clusters,
              fill='toself',
              name=index
        ))
    return fig

@st.cache
def get_explained_variance_plot(expenditures_scaled):
    '''Gets scaled expenditures and creates a plot for explained variance'''
    pca = PCA()
    pca.fit_transform(expenditures_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x = range(1, explained_variance.shape[0] + 1),
        y = explained_variance,
        labels = {"x": "Number of Principal Components", "y": "Explained Variance"}
    )
    return fig