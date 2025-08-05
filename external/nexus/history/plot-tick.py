import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nexus.charting import show_plotly_figure

def main():
    # Check if the correct number of arguments have been provided
    if len(sys.argv) != 2:
        print("Usage: py chart.py <csv_file>")
        print("Example: py chart.py AAPL.csv")
        sys.exit(1)

    # Parse command-line argument
    csv_file = sys.argv[1]

    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Create a subplot with 2 rows, the first for the OHLC chart and the second for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('Ticks', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Add Ticks chart to the first row

    fig.add_trace(go.Scatter(
            x=data['<TIME>'],
            y=data['<LAST>'],
            mode='lines',
            name='Ticks'
        ), row=1, col=1)
    
    # Add volume bar chart to the second row
    fig.add_trace(go.Bar(
        x=data['<TIME>'],
        y=data['<VOL>'],
        name='Volume'
    ), row=2, col=1)

    # Update layout for better visualization
    fig.update_layout(
        title=f'Ticks and Volume Chart for {csv_file}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis2_title='Date',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    # Show the chart
    show_plotly_figure(fig) # WSL workaround, instead of fig.show()

if __name__ == "__main__":
    main()