import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nexus.charting import show_plotly_figure

from nexus.indicators.utils import push, genSine
from nexus.indicators.laguerre import LaguerreFilter
from nexus.indicators.bandpass import BandPass
from nexus.logger import setup_logging

setup_logging("plot_filters")
logger = logging.getLogger("demo")     

# Simulation parameters
num_bars = 800  # Total number of bars
look_back = 100  # Look-back period
period1 = 3  # Starting period for sine/square wave
period2 = 60  # Ending period for sine/square wave

# Timeseries to hold generated waveforms
chirp_series = np.zeros(num_bars-look_back)
laguerre_series = np.zeros(num_bars-look_back)
bandpass_series = np.zeros(num_bars-look_back)

# Init Filters
laguerre = LaguerreFilter(alpha=0.3)
bandpass = BandPass(30,0.1)

# Generate the series by looping over the bars
for nBar in range(1,num_bars):
    chirp = genSine(period1, period2, num_bars, look_back, nBar)
    logger.info(f"<<chirp,{nBar},{chirp:.8f}>>")  
    push(chirp_series, chirp)
    push(laguerre_series, laguerre.update(chirp_series))
    push(bandpass_series, bandpass.update(chirp_series))

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Chirp", "Laguerre Filter", "BandPass Filter"),
    shared_xaxes=True,
    vertical_spacing=0.05
)

# Add sine wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=chirp_series[::-1], mode='lines', name='Chirp source'),
      row=1, col=1
    )

# Add square wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=laguerre_series[::-1], mode='lines', name='Laguerre'),
    row=2, col=1
    )

# Add noise wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=bandpass_series[::-1], mode='lines', name='BandPass'),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=900,
    width=1200,
    title_text="Filters Demo",
    showlegend=False
)

# Update axes labels
fig.update_xaxes(title_text="Sample Number", row=3, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=2, col=1)
fig.update_yaxes(title_text="Amplitude", row=3, col=1)

# Show the plot
show_plotly_figure(fig) # WSL workaround, instead of fig.show()
