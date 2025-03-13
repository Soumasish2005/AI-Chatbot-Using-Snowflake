import streamlit as st
import pandas as pd
import snowflake.connector
import google.generativeai as genai
import matplotlib.pyplot as plt
import re
import seaborn as sns
# 🔹 Google Gemini API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 Snowflake Connection Config
SNOWFLAKE_CONFIG = {
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
    "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"],
    "database": st.secrets["SNOWFLAKE_DATABASE"],
    "schema": st.secrets["SNOWFLAKE_SCHEMA"],
}

def connect_snowflake():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

def get_table_columns(conn, table_name):
    cursor = conn.cursor()
    try:
        cursor.execute(f"DESC TABLE {table_name}")
        return [row[0].upper() for row in cursor.fetchall()]
    finally:
        cursor.close()

def create_table_if_not_exists(conn, table_name, df):
    cursor = conn.cursor()
    df.columns = df.columns.str.upper()
    column_defs = ", ".join([f'"{col}" STRING' for col in df.columns])
    cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_defs});')
    cursor.close()

def insert_data_to_snowflake(conn, table_name, df):
    cursor = conn.cursor()
    df.columns = [col.upper() for col in df.columns]
    df = df.where(pd.notna(df), None)
    df = df.astype(str).replace("None", None)
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    if not cursor.fetchone():
        column_defs = ", ".join([f'"{col}" STRING' for col in df.columns])
        cursor.execute(f'CREATE TABLE {table_name} ({column_defs});')
    cursor.execute(f"DESC TABLE {table_name}")
    snowflake_columns = [row[0].upper() for row in cursor.fetchall()]
    missing_cols = [col for col in df.columns if col not in snowflake_columns]
    if missing_cols:
        raise ValueError(f"❌ Missing columns {missing_cols} in {table_name}")
    placeholders = ", ".join(["%s"] * len(df.columns))  
    sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({placeholders})"
    cursor.executemany(sql, df.itertuples(index=False, name=None))
    conn.commit()
    cursor.close()

def generate_sql_query(user_query, table_name, conn):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    sample_rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    cursor.close()
    sample_data = "\n".join([", ".join(map(str, row)) for row in sample_rows])
    formatted_table = f"Columns: {', '.join(col_names)}\nSample Data:\n{sample_data}"
    prompt = f"""
            Convert the following natural language question into a valid Snowflake SQL query:

            - Table name: `{table_name}`
            - Dataset structure (column names and sample values):
            {formatted_table}
            - Ensure valid Snowflake SQL syntax.
            - If extracting YEAR, MONTH, or DAY from a date column, first ensure it's converted using TO_DATE(column_name, 'YYYY-MM-DD') if stored as VARCHAR.
            - If performing numerical operations, use TO_NUMBER(column_name) if the column is stored as VARCHAR.
            - Match column values exactly as they appear in the dataset.
            - Do not wrap the query in markdown formatting.

            Question: "{user_query}"
            """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    sql_query = re.sub(r"```sql|```", "", response.text.strip()).strip()
    return sql_query if sql_query.upper().startswith("SELECT") else f'SELECT * FROM "{table_name}" WHERE {sql_query};'

def execute_sql_query(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    finally:
        cursor.close()

def plot_data(df):
    if df.empty:
        st.write("⚠️ No data to visualize.")
        return
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(num_cols) == 1 and len(cat_cols) == 1:
        df.columns = ["Category", "Value"]
        sns.barplot(x="Category", y="Value", data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    elif len(num_cols) >= 2:
        df.plot(kind="line", ax=ax, marker="o")
        plt.legend(title="Metrics")
    elif len(num_cols) == 1:
        sns.histplot(df[num_cols[0]], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

st.title("Conversational Data Analysis Bot 📊")
st.write("Upload a CSV, and ask questions about your data!")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    table_name = "ANALYSIS_DATA"
    conn = connect_snowflake()
    with st.spinner("Setting up database..."):
        create_table_if_not_exists(conn, table_name, df)
        insert_data_to_snowflake(conn, table_name, df)
        st.success("✅ Data uploaded successfully!")
    user_query = st.text_input("Ask a question about your data:")
    if user_query:
        with st.spinner("Generating SQL query..."):
            sql_query = generate_sql_query(user_query, table_name, conn)
            st.code(sql_query, language="sql")
        with st.spinner("Fetching results..."):
            result_df = execute_sql_query(conn, sql_query)
            st.dataframe(result_df)
        if not result_df.empty:
            plot_data(result_df)
    conn.close()
