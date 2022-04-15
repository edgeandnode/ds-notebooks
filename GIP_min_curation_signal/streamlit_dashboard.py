import streamlit as st
import pandas as pd
import numpy as np
import sympy
from sympy import *
from PIL import Image
import seaborn as sns
import requests
import json
import pathlib

# supress scientific notation
pd.options.display.float_format = '{:.5f}'.format

# function to query data to convert X to curation signal
@st.cache(allow_output_mutation=True)
def query_stats():
    query = '''{
    graphNetworks{
        totalTokensSignalled
        totalTokensAllocated
        totalSupply
        epochLength
        networkGRTIssuance
    }
    }'''
    # set endpoint url
    url = 'https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet'
    # make the request
    r = requests.post(url, json={'query': query})
    # load result into json
    json_data = json.loads(r.text)
    # convert json into a dataframe
    df = pd.DataFrame(json_data['data']['graphNetworks'])
    # Make adjustments
    df['totalTokensSignalled'] = df['totalTokensSignalled'].astype('float').div(10**18)
    df['totalTokensAllocated'] = df['totalTokensAllocated'].astype('float').div(10**18)
    df['totalSupply'] = df['totalSupply'].astype('float').div(10**18)
    df['networkGRTIssuance'] = df['networkGRTIssuance'].astype('float').div(10**18)
    return df
# pull data
network_stats = query_stats()

# Define equations #
# initialize symbols for equations
init_printing()
X, c_O, c_I, r_i, l, Δ, R_i, σ, ψ, τ_d, D  = symbols('X c_O c_I r_i l Δ, R_i, σ, ψ, τ_d, D')

def fun25():
    # calculate delta
    return Piecewise((σ*ψ, σ*ψ < τ_d/(1-τ_d)), ((σ*ψ)/(1+D) + (τ_d*D)/((1-τ_d)*(1+D)), σ*ψ >= τ_d/(1-τ_d)))
# define equation 23
def fun23(X):
    # calculate new delta
    delta = fun25()
    # get results of equation 23
    return 2 * ((sqrt(X)-sqrt(c_O+c_I))/sqrt(r_i*l)) * sqrt(X) * (sqrt(r_i*l+delta)-sqrt(r_i*l)) - ((sqrt(X)-sqrt(c_O+c_I))**2)/(r_i*l) * delta - c_O

# Title
st.markdown("# Prysm Followups")
# Section
st.markdown("## Equations overview")
# Show equation 25 
st.markdown("### Equation 25")
st.image(Image.open('/app/prysmfollowups2/streamlit/images/eq_25.png'), caption='delta (Eq 25) used in Eq 23')
st.write("Translated to Python code:")
st.text(fun25())
st.write("Python code translated to LaTeX:")
st.latex(fun25())
# Equation 23
st.markdown("### Equation 23")
# Show equation 23 
st.image(Image.open('/app/prysmfollowups2/streamlit/images/eq_23.png'), caption='Equation 23')
st.write("Translated to Python code:")
st.text(fun23(X))
st.write("Python code translated to LaTeX:")
st.latex(fun23(X))

# Define user inputs/sliders #
st.sidebar.markdown('### Choose fixed parameters for heatmaps')
l = st.sidebar.slider("l - length of allocation (fraction of a year)", 0.0, 50/365, 28/365, 1/365)
r_i = st.sidebar.slider("r_i - annual opportunity cost of indexer", 0.01, 0.4, 0.2, 0.01)
#σ = st.sidebar.slider("σ - probability of attacker being detected", 0.0, 1.0, 0.1, 0.1)
#ψ = st.sidebar.slider("ψ - slashing percentage", 0.0, 1.0, 0.025, 0.01)
τ_d = st.sidebar.slider("τ_d - delegation tax", 0.0, 0.3, 0.005, 0.001)
D = st.sidebar.slider("D - max delegation ratio", 1, 32, 16, 1)
c_I = st.sidebar.slider("c_I - cost of indexing subgraph", 1, 2000, 500, 10)
c_O = st.sidebar.slider("c_O - cost of serving queries", 0, 1000, 10, 10)
st.sidebar.markdown('### Filters to adjust conversion of X to curation signal')
GRT_USD = st.sidebar.slider("USD Price of GRT", 0.0, 10.0, 0.45, 0.05)
GRT_inflation = st.sidebar.slider("Current inflation of GRT distributed as indexing rewards", 2.0, 3.0, 2.7, 0.1)

# HEATMAP - EQ 23#
st.markdown("## Equation 23 - min X to satisfy zero-profit condition")
# initialize arrays for heatmap
results_data = pd.DataFrame([])
# iterate over values
for σ in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    #print(i)
    for ψ in [0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1]:
        # calculate new result for equation 23
        result_eq23 = solveset(fun23(X), X)
        # get min of equation 24 results
        result_eq23_min = min(result_eq23)
        # store minimum result of either equation in dataframe
        results_data = results_data.append(pd.DataFrame({'σ': σ, 'ψ': ψ, 'x_min': int(result_eq23_min)}, index=[0]), ignore_index=True)

#pivot data
results_data_pivot = results_data.pivot("σ", "ψ", "x_min")
# show data
sns.heatmap(results_data_pivot, annot=True,fmt='g', cmap="rocket_r")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Convert heatmap of min X to heatmap of minimum signal #
# Convert network stats to curation signal
st.markdown("### Convert X to curation signal")
network_stats['GRTIssuancePerEpoch'] = 10000000000 * (GRT_inflation/100)/365 #results_data['networkGRTIssuance']**results_data['epochLength']
network_stats['GRT_USD'] = GRT_USD
network_stats['USDIssuancePerEpoch'] = network_stats['GRT_USD']*network_stats['GRTIssuancePerEpoch']
# Calculate USD issuance per token signalled
network_stats['USD_issuance_by_token_signalled'] = network_stats['USDIssuancePerEpoch']/network_stats['totalTokensSignalled']
# show results
#st.write("The following data was pulled using the mainnet subgraph for this calculation (there is a slider for price of GRT in USD):")
#st.write(network_stats)
#st.markdown('Steps to convert X to curation signal: \n 1. Pull subgraph data - total signal, GRT issuance per block, epoch length \n 2. Calculate the GRT issued as indexing rewards per epoch \n 3. Convert new GRT issuance to USD (using manual filter on sidebar to set USD price) \n 4. Find USD indexing rewards for each token in curation signal \n 5. Divide X (indexing reward on subgraph in USD) by USD rewards by token signalled over time period to find the curation signal associated with X')

st.markdown("### Equation 23 - Minimum Curation Signal")
results_data['curation_min'] = results_data['x_min']/(network_stats['USD_issuance_by_token_signalled'][0]*(l*365))
# convert to int
results_data['curation_min'].astype('int')
#pivot data
results_data_pivot = results_data.pivot("σ", "ψ", "curation_min")
# show data
sns.heatmap(results_data_pivot, annot=True,fmt='g', cmap="rocket_r", annot_kws={"size": 5})
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
