import asyncio
import json
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from core.backtesting.optimizer import StrategyOptimizer
from core.services.backend_api_client import BackendAPIClient
from core.services.timescale_client import TimescaleClient

st.set_page_config(layout="wide", page_title="Quants-View")


@st.cache_data
def get_study_names(_optimizer: StrategyOptimizer) -> List[str]:
    return _optimizer.get_all_study_names()


@st.cache_data
def get_study_trials_df_dict(_optimizer: StrategyOptimizer, study_names: List[str]) -> Dict[str, pd.DataFrame]:
    study_trials_df_dict = {}
    for study in study_names:
        df = _optimizer.get_study_trials_df(study)
        study_trials_df_dict[study] = preprocess_trials_df(df)
    return study_trials_df_dict


def preprocess_trials_df(trials_df):
    df = trials_df.copy()
    df.columns = [col.replace("params_", "") if col.startswith("params_") else col for col in df.columns]
    df["executors"] = df["executors"].apply(lambda x: json.loads(x))
    df["config"] = df["config"].apply(lambda x: json.loads(x))
    df["from_timestamp"] = df["executors"].apply(
        lambda x: pd.to_datetime(pd.Series(x["timestamp"].values()).min(), unit="s"))
    df["to_timestamp"] = df["executors"].apply(
        lambda x: pd.to_datetime(pd.Series(x["close_timestamp"].values()).max(), unit="s"))
    return df


def study_trial_summary_gantt(df: pd.DataFrame):
    df['datetime_start'] = pd.to_datetime(df['datetime_start'])
    df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])

    # Create the Gantt chart using Plotly Express
    fig = px.timeline(
        df,
        x_start="datetime_start",
        x_end="datetime_complete",
        y="study_trial",
        hover_name="study_trial",
        hover_data={
            "number_of_trials": True,
            "datetime_start": False,
            "datetime_complete": False,
            "net_pnl_summary": True,
            "sharpe_ratio_summary": True
        },
        title="Gantt Chart of Study Trials"
    )

    # Customize the hover template to show net_pnl_summary and sharpe_ratio_summary
    fig.update_traces(
        hovertemplate=(
            "Study Trial: %{y}<br>"
            "Number of Trials: %{customdata[0]}<br>"
            "Net PNL Summary: %{customdata[1]}<br>"
            "Sharpe Ratio Summary: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )

    # Adjust layout for better visualization
    fig.update_layout(
        xaxis_title="Date and Time",
        yaxis_title="Study Trials",
        xaxis=dict(tickformat="%Y-%m-%d %H:%M", ),
        yaxis=dict(autorange="reversed"),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        height=600
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def calculate_backtesting_time_spent(df: pd.DataFrame):
    summary_df = df.sort_values(by="datetime_start").reset_index(drop=True).copy()

    total_inactive_time = pd.Timedelta(0)
    previous_end_time = None

    for _, row in summary_df.iterrows():
        if previous_end_time is not None:
            gap = row["datetime_start"] - previous_end_time
            if gap > pd.Timedelta(0):  # Only add if there's a positive gap (no overlap)
                total_inactive_time += gap

        previous_end_time = max(previous_end_time, row["datetime_complete"]) if previous_end_time else row[
            "datetime_complete"]

    # Results
    return total_inactive_time.total_seconds()


def get_all_trials_df(studies_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_trials_df = pd.DataFrame()
    for study_name, trial_df in studies_dict.items():
        trial_df["study_name"] = study_name
        all_trials_df = pd.concat([all_trials_df, trial_df])
    return all_trials_df


def download_csv_button(df: pd.DataFrame, file_name: str):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV",
                       data=csv,
                       file_name=f"{file_name}.csv",
                       mime="text/csv")


def generate_top_markets_report(metrics_df: pd.DataFrame, volatility_threshold: float,
                                volume_threshold: float, max_top_markets: int = 50):
    metrics_df.sort_values(by=['trading_pair', 'to_timestamp'], ascending=[True, False])
    screener_report = metrics_df.drop_duplicates(subset='trading_pair', keep='first')
    screener_report.sort_values("mean_natr", ascending=False, inplace=True)
    natr_percentile = screener_report['mean_natr'].astype(float).quantile(volatility_threshold)
    volume_percentile = screener_report['average_volume_per_hour'].astype(float).quantile(volume_threshold)
    screener_top_markets = screener_report[
        (screener_report['mean_natr'] > natr_percentile) &
        (screener_report['average_volume_per_hour'] > volume_percentile)].head(max_top_markets)
    screener_top_markets["market"] = screener_top_markets["connector_name"] + "_" + screener_top_markets[
        "trading_pair"]
    screener_top_markets.sort_values("to_timestamp", ascending=False, inplace=True)
    return screener_top_markets


def get_screener_top_markets_gantt_fig(df: pd.DataFrame):
    fig = px.timeline(
        df,
        x_start="from_timestamp",
        x_end="to_timestamp",
        y="market",
        title="Gantt Chart of Market Activity",
        labels={"market": "Market"}
    )

    # Customize layout for readability
    fig.update_layout(
        xaxis_title="Date and Time",
        yaxis_title="Market",
        xaxis=dict(tickformat="%Y-%m-%d %H:%M"),
        yaxis=dict(autorange="reversed"),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        height=600
    )
    return fig


async def main():
    st.title("Welcome to Quants-View")
    ts_client = TimescaleClient(host="localhost")
    await ts_client.connect()
    optimizer = StrategyOptimizer(engine="postgres",
                                  db_client=ts_client,
                                  db_host="localhost",
                                  db_user="admin",
                                  db_pass="admin",
                                  db_port=5433)
    backend_api_client = BackendAPIClient(host="localhost")

    # Define the variables to initialize and their respective functions
    initial_vars = {
        "study_trials": lambda: get_study_names(optimizer),
        "study_trials_df_dict": lambda: get_study_trials_df_dict(optimizer, st.session_state["study_trials"])
    }

    # Loop through each variable to initialize
    for var_name, init_func in initial_vars.items():
        if var_name not in st.session_state:
            st.session_state[var_name] = init_func()

    trials_analysis_tab, db_status_tab, audit_tab = st.tabs(["Trials Analysis", "DB Status", "Audit"])
    with trials_analysis_tab:
        st.subheader("Trial Analysis")
        trials_df = get_all_trials_df(st.session_state.study_trials_df_dict)

        col1, col2, col3 = st.columns(3)
        min_sharpe_ratio = col1.number_input("Min Sharpe Ratio", 0.0)
        min_total_positions = col2.number_input("Min Total Positions", 1)
        min_net_pnl_pct = col3.number_input("Min Net Pnl %", 0.0)

        filtered_trials = trials_df[(trials_df["sharpe_ratio"] >= min_sharpe_ratio) &
                                    (trials_df["total_positions"] >= min_total_positions) &
                                    (trials_df["net_pnl"] >= min_net_pnl_pct)]
        sharpe_vs_total_positions_fig = px.scatter(filtered_trials, x="total_executors", y="sharpe_ratio",
                                                   color="net_pnl",
                                                   hover_data=["study_name", "number", "interval", "total_volume"])
        net_pnl_vs_max_drawdown_fig = px.scatter(filtered_trials, x="max_drawdown_pct", y="net_pnl",
                                                 color="total_positions",
                                                 hover_data=["study_name", "number", "interval",
                                                             "sharpe_ratio", "total_volume"])
        col1, col2 = st.columns(2)
        col1.plotly_chart(sharpe_vs_total_positions_fig, use_container_width=True)
        col2.plotly_chart(net_pnl_vs_max_drawdown_fig, use_container_width=True)
        executors_df = pd.DataFrame()
        for _, row in filtered_trials.iterrows():
            df = pd.DataFrame(row["executors"])
            cols_to_add = ["number", "sharpe_ratio", "study_name", "total_positions", "net_pnl"]
            for col in cols_to_add:
                df[col] = row[col]
            config_cols_to_add = ["trading_pair"]
            for col in config_cols_to_add:
                df[col] = row["config"][col]
            executors_df = pd.concat([executors_df, df])
        trial_performance = executors_df.sort_values(by="close_timestamp")
        trial_performance = (
            trial_performance.groupby(
                ["study_name", "number", "trading_pair", "sharpe_ratio", "net_pnl", "total_positions"])
            .apply(lambda x: pd.Series({
                "cum_net_pnl_quote": x["net_pnl_quote"].cumsum().tolist(),
                "data_range": (pd.to_datetime(x["timestamp"], unit="s").min().strftime("%Y-%m-%d %H:%M:%S"),
                               pd.to_datetime(x["close_timestamp"], unit="s").max().strftime("%Y-%m-%d %H:%M:%S"))
            }))
            .reset_index()
        )
        trial_performance["net_pnl"] = trial_performance["net_pnl"].apply(lambda x: f"{x * 100:.2f}%")

        st.data_editor(trial_performance,
                       hide_index=True,
                       column_config={
                           "cum_net_pnl_quote": st.column_config.LineChartColumn("Cum. Net Pnl Quote", width=400),
                       },
                       use_container_width=True)
        st.subheader("Select your favorite trial")
        c1, c2, c3 = st.columns([1, 1, 0.5])
        with c1:
            study_name = st.selectbox("Study Name", trial_performance["study_name"].unique())
        with c2:
            trial_numbers = filtered_trials.loc[filtered_trials["study_name"] == study_name, "number"].unique()
            trial_number = st.selectbox("Trial Number", trial_numbers)
        config = filtered_trials.loc[(filtered_trials["study_name"] == study_name) &
                                     (filtered_trials["number"] == trial_number), "config"].iloc[0].copy()
        with c1:
            config_id = st.text_input("Controller ID", config["id"])
        with c2:
            version = st.text_input("Select version", "0.1")
            config["id"] = config_id + "_" + version
        with c3:
            backtest_button = st.button("Backtest")
            save_config_button = st.button("Upload to Backend API")

        if backtest_button:
            # NANUPO MOSTRA EL BT
            pass
        if save_config_button:
            msg = await backend_api_client.add_controller_config(config)
            st.info(msg)

        st.json(config, expanded=False)

    with db_status_tab:
        col1, col2, col3 = st.columns(3)
        metrics_df = await ts_client.get_metrics_df()
        volatility_threshold = col1.number_input("Volatility Percentile Threshold", 0.0)
        volume_threshold = col2.number_input("Volume Percentile Threshold", 0.0)
        max_top_markets = col3.number_input("Max Top Markets", 10)
        screener_top_markets = generate_top_markets_report(metrics_df=metrics_df,
                                                           volatility_threshold=volatility_threshold,
                                                           volume_threshold=volume_threshold,
                                                           max_top_markets=max_top_markets)
        screener_top_markets_gantt_fig = get_screener_top_markets_gantt_fig(screener_top_markets)
        st.plotly_chart(screener_top_markets_gantt_fig, use_container_width=True)

    with audit_tab:
        st.subheader("Audit")

        # Initialize an empty list to store summary data
        summary_data = []

        # Loop through each key-value pair in the dictionary of DataFrames
        for study_trial, df in st.session_state.study_trials_df_dict.items():
            number_of_trials = len(df)
            datetime_start = df['datetime_start'].min()
            datetime_complete = df['datetime_complete'].max()
            max_net_pnl = df['net_pnl'].max()
            max_sharpe_ratio = df['sharpe_ratio'].max()

            # Construct custom strings
            net_pnl_summary = f"{number_of_trials} || {max_net_pnl}"
            sharpe_ratio_summary = f"{number_of_trials} || {max_sharpe_ratio}"

            # Append a dictionary of summary info for the current trial to the list
            summary_data.append({
                "study_trial": study_trial,
                "number_of_trials": number_of_trials,
                "datetime_start": datetime_start,
                "datetime_complete": datetime_complete,
                "net_pnl_summary": net_pnl_summary,
                "sharpe_ratio_summary": sharpe_ratio_summary
            })

        # Convert the summary data list into a DataFrame
        summary_df = pd.DataFrame(summary_data)
        total_time = (summary_df["datetime_complete"].max() - summary_df["datetime_start"].min()).total_seconds()
        inactive_time = calculate_backtesting_time_spent(summary_df)
        activity_pct = (total_time - inactive_time) / total_time

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("% Activity", f"{100 * activity_pct:.2f} %")
        col2.metric("Total Days", f"{total_time / (60 * 60 * 24):.2f}")
        col3.metric("# Studies", len(summary_df))
        if activity_pct < 0.5:
            col4.warning("Sólido más de la mitad del tiempo pelotudeando")
        elif activity_pct < 0.7:
            col4.warning("Hay búsqueda pero poco encuentro")
        elif activity_pct < 0.9:
            col4.info("Está planteado")
        elif activity_pct < 0.99:
            col4.success("Hay búsqueda y también encuentro")
        else:
            st.balloons()
            col4.success("Fucking legend")
        study_trial_summary_gantt(summary_df)

    st.subheader("Tables")
    with st.expander("Trials", expanded=False):
        st.dataframe(filtered_trials, use_container_width=True)
        download_csv_button(filtered_trials, "filtered_trials")

    with st.expander("Executors", expanded=False):
        st.dataframe(executors_df, use_container_width=True)
        download_csv_button(executors_df, "all_executors")


if __name__ == "__main__":
    asyncio.run(main())
