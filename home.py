import streamlit as st

st.title("📺 TV Program Scheduling Optimizer")
st.write("This app uses a Genetic Algorithm to find the best TV schedule based on ratings.")

uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv file", type="csv")

if uploaded_file:
    program_ratings_dict = read_csv_to_dict(uploaded_file.name)

    # Save uploaded file temporarily
    with open("uploaded_file.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    program_ratings_dict = read_csv_to_dict("uploaded_file.csv")

    if not program_ratings_dict:
        st.warning("⚠️ No data found in the uploaded CSV.")
        st.stop()
