

path = '/home/shaswati/Documents/PSF/60x-Si-1.3-actual-banana/plots/radial_profiles_psf_60x_si_bague0.json'

import json
import plotly.graph_objs as go

def plot_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    # Preparing data for the graph
    x_values = range(len(data['1'])) # The abscissa are simply the rank of each ordinate 
    y_values = [data[key] for key in data.keys()]

    #Creating the graph
    fig = go.Figure()
    for (key, y_values) in data.items():
        fig.add_trace(go.Scatter(x=list(x_values), y=y_values, mode='lines', name=f"Clé {key}"))

    fig.update_layout(
        title='Graph of ordered lists',
        xaxis_title='Ordinate Index',
        yaxis_title='Value',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()
