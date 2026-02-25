# Healthcare Analytics with Forecast Dashboard

An end-to-end data analytics and predictive modeling platform designed to optimize hospital operations, visualize patient demographics, and forecast daily admissions. This project combines Python-based machine learning, a dynamic web application, and Business Intelligence (BI) reporting to provide actionable insights for healthcare management.

## Key Features

* **Data Engineering & Cleaning:** Raw hospital datasets (appointments and patient records) were rigorously cleaned, transformed, and preprocessed using Python in **Jupyter Notebooks**.
* **Interactive Web Application:** A fully responsive front-end dashboard built with **Streamlit**, featuring secure role-based login and a custom animated UI.
* **Predictive Analytics (ARIMA):** Integrated Time Series Forecasting using the `statsmodels` ARIMA algorithm to predict future daily hospital admissions based on historical trends.
* **Context-Aware AI Assistant ("Clara"):** A custom AI chatbot powered by the **Google Gemini API**. Clara reads the active statistical summaries of the hospital datasets and answers user queries in real-time.
* **Business Intelligence Dashboard:** A separate, highly interactive **Power BI** dashboard created to provide executives with drill-down capabilities into department metrics, bed occupancy, and patient distributions.
* **Local Database Integration:** Automated generation of a local **SQLite** database to store and query cleaned records seamlessly.

##  Technology Stack 🛠️

* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy, Jupyter Notebook
* **Machine Learning & Stats:** Statsmodels (ARIMA)
* **Web Framework:** Streamlit
* **Data Visualization:** Plotly, Matplotlib, Seaborn, Power BI
* **Database:** SQLite3
* **Generative AI:** Google Generative AI SDK (Gemini)

## Project Structure 📁 

```text
Hospital-Analytics-Project/
├── dataset/                     # Contains raw and cleaned CSV files
│   ├── appointments_cleaned.csv
│   └── patients_cleaned.csv
├── notebooks/                   # Jupyter Notebooks for EDA and Data Cleaning
│   └── data_preprocessing.ipynb
├── powerbi/                     # Power BI dashboard files (.pbix)
│   └── healthcare_dashboard.pbix
├── app.py                       # Main Streamlit application script
├── imggifproject.gif            # Animated UI asset for the login page
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

Installation & Local Setup 🚀 
To run this application locally on your machine, follow these steps:

1. Clone the repository

Bash
git clone [https://github.com/yourusername/Hospital-Analytics-Project.git](https://github.com/yourusername/Hospital-Analytics-Project.git)
cd Hospital-Analytics-Project
2. Create a virtual environment (Optional but recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install the required dependencies

Bash
pip install -r requirements.txt
4. Configure the AI API Key
Create a hidden .streamlit folder in the root directory and add a secrets.toml file to securely store your Google Gemini API key:

Ini, TOML
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_api_key_here"
5. Launch the application

Bash
streamlit run app.py

Usage & Access 🔐 

When the application launches, you will be greeted by the secure login portal.
Username: admin
Password: admin123

Select your target hospital node to initialize the data and access the analytics, visualizations, and the AI chatbot.

Authors 👨‍💻
Insha Farhan & Diksha Tiwari

📄 License:
This project is developed for educational and academic purposes. Free to use and modify with proper attribution.

⭐ If you like this project, give it a star on GitHub!
📧 For queries: inshafarhan55@gmail.com
tdiksha0408@gmail.com

