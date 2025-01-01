import streamlit as st

from dashboards.f1_score_review import main as f1_dashboard
from dashboards.key_correlation_review import main as key_dashboard
from dashboards.dstart_correlation_review import main as dstart_dashboard
from dashboards.duration_correlation_review import main as duration_dashboard
from dashboards.velocity_correlation_review import main as velocity_dashboard
from dashboards.pitch_correlation_review import main as pitch_distribution_dashboard

DASHBOARDS = {
    "Pitch Distribution Analysis": pitch_distribution_dashboard,
    "Velocity Distribution Analysis": velocity_dashboard,
    "Dstart Distribution Analysis": dstart_dashboard,
    "Duration Distribution Analysis": duration_dashboard,
    "F1 Score Analysis": f1_dashboard,
    "Key Distribution Analysis": key_dashboard,
}


def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("MIDI Analysis Dashboards")
    selected_dashboard = st.sidebar.selectbox(
        label="Select Analysis Type",
        options=list(DASHBOARDS.keys()),
    )

    DASHBOARDS[selected_dashboard]()


if __name__ == "__main__":
    main()
