import streamlit as st
import pandas as pd
from ga_tv_scheduler import read_ratings, evolve, schedule_to_table, HOUR_COLUMNS  # adjust imports as needed
# OR copy the evolve/read functions into this file

st.title("GA TV Scheduler")

uploaded = st.file_uploader("Upload program_ratings.csv (first column program name, columns Hour 6..Hour 23)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head())

    # Parameters for three trials
    st.sidebar.header("Trials parameters")
    st.sidebar.write("Trial 1")
    co1 = st.sidebar.slider("CO_R trial1", 0.0, 0.95, 0.8)
    mu1 = st.sidebar.slider("MUT_R trial1", 0.01, 0.05, 0.02)
    st.sidebar.write("Trial 2")
    co2 = st.sidebar.slider("CO_R trial2", 0.0, 0.95, 0.9)
    mu2 = st.sidebar.slider("MUT_R trial2", 0.01, 0.05, 0.03)
    st.sidebar.write("Trial 3")
    co3 = st.sidebar.slider("CO_R trial3", 0.0, 0.95, 0.7)
    mu3 = st.sidebar.slider("MUT_R trial3", 0.01, 0.05, 0.01)

    if st.button("Run 3 Trials"):
        # save uploaded temporarily
        uploaded_path = "uploaded_program_ratings.csv"
        df.to_csv(uploaded_path, index=False)
        param_sets = [(co1, mu1), (co2, mu2), (co3, mu3)]
        # Call the run function (imported) - for streamlit, consider showing progress
        results = run_three_trials(uploaded_path, param_sets)
        for r in results:
            st.subheader(f"Trial {r['trial']} (CO_R={r['co_r']}, MUT_R={r['mut_r']})")
            st.table(r['schedule_df'])
            st.write("Total Fitness:", round(r['score'],4))
