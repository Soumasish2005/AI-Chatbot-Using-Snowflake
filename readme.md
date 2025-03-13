# Conversational Data Analysis Bot 📊

This project is a Streamlit application that allows users to upload a CSV file and ask questions about their data in natural language. The application uses Google Gemini API to generate SQL queries from user questions and executes them on a Snowflake database. The results are then visualized using Matplotlib and Seaborn.

## Features

- Upload CSV files and preview the data.
- Ask questions about the data in natural language.
- Automatically generate and execute SQL queries on a Snowflake database.
- Visualize the results using various chart types.

## Setup

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Pandas
- Snowflake Connector for Python
- Google Generative AI (Gemini API)
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Soumasish2005/conversational-data-analysis-bot.git
    cd conversational-data-analysis-bot
    ```

2. Install the required packages:
    ```sh
    pip install streamlit pandas snowflake-connector-python google-generativeai matplotlib seaborn
    ```

3. Configure your Google Gemini API key and Snowflake connection settings in `.streamlit/secrets.toml`:
    ```toml
    # filepath: ./.streamlit/secrets.toml
    GEMINI_API_KEY = "your_google_gemini_api_key"
    SNOWFLAKE_USER = "your_snowflake_user"
    SNOWFLAKE_PASSWORD = "your_snowflake_password"
    SNOWFLAKE_ACCOUNT = "your_snowflake_account"
    SNOWFLAKE_WAREHOUSE = "your_snowflake_warehouse"
    SNOWFLAKE_DATABASE = "your_snowflake_database"
    SNOWFLAKE_SCHEMA = "your_snowflake_schema"
    ```

### Running the Application

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the application.

## Usage

1. Upload a CSV file using the file uploader.
2. Preview the uploaded data.
3. Ask questions about the data in the text input field.
4. View the generated SQL query and the results.
5. Visualize the data using the provided charts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Snowflake](https://www.snowflake.com/)
- [Google Generative AI](https://cloud.google.com/generative-ai)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
