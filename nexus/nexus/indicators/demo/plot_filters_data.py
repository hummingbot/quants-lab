"""
 Minimal example using BTCUSDT data
"""
import plotly.graph_objects as go
from nexus.feed.klines import read_klines
from nexus.indicators.laguerre import LaguerreFilter
from nexus.indicators.ehlers import SuperSmoother, UltimateSmoother
from nexus.charting import show_plotly_figure

def main():
    df = read_klines("history/binance-futures/BTCUSDT_1m.parquet")  
    df = df.head(1000)  # Use a subset for demonstration

    laguerre_filter = LaguerreFilter(alpha=0.5)
    super_smoother_filter = SuperSmoother(cutoff=20)
    ultimate_smoother_filter = UltimateSmoother(length=20)

    laguerre = []
    super_smoother = []
    ultimate_smoother = []

    for price in df['close']:
        laguerre.append(laguerre_filter.update(price))
        super_smoother.append(super_smoother_filter.update(price))
        ultimate_smoother.append(ultimate_smoother_filter.update(price))

    df['laguerre'] = laguerre
    df['super_smoother'] = super_smoother
    df['ultimate_smoother'] = ultimate_smoother

    # Plot original and smoothed prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['open_time'], y=df['close'],
                             mode='lines', name='Price (Close)'))
    fig.add_trace(go.Scatter(x=df['open_time'], y=df['laguerre'],
                             mode='lines', name='LaguerreFilter'))
    fig.add_trace(go.Scatter(x=df['open_time'], y=df['super_smoother'],
                             mode='lines', name='SuperSmoother'))
    fig.add_trace(go.Scatter(x=df['open_time'], y=df['ultimate_smoother'],
                             mode='lines', name='UltimateSmoother'))
    fig.update_layout(title='BTCUSDT Close Prices - Various Filter in Action',
                      xaxis_title='Time',
                      yaxis_title='Price (USDT)')

    show_plotly_figure(fig) # WSL workaround, instead of fig.show()

if __name__ == "__main__":
    main()
