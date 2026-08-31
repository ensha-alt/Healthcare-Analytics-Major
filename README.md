# 🏥 CareCast: Hospital Admission and Bed Occupancy Analysis with Forecast

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboards-F2C811.svg)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)
![AI Chatbot](https://img.shields.io/badge/AI-LLM_Integrated-8A2BE2.svg)

## 📌 Project Overview
**CareCast** is an end-to-end healthcare analytics and forecasting system designed to solve critical operational challenges in hospital capacity management. By leveraging historical admission data, advanced time-series forecasting (ARIMA & Prophet), and interactive dashboards, this project enables proactive, data-driven decision-making for hospital administrators. 

A standout feature of CareCast is the integration of an **LLM-powered conversational AI**, allowing non-technical hospital staff to query complex data insights using natural language.

## ✨ Key Features
* **Predictive Capacity Planning:** Forecasts future patient inflow and bed occupancy using robust time-series models.
* **Dual-Platform Visualization:** * **Power BI** for high-level, executive Business Intelligence reporting.
  * **Streamlit** for an accessible, interactive, and predictive web application.
* **Intelligent Querying:** An embedded AI chatbot (powered by an external LLM API) that provides instant, context-aware answers to user queries regarding hospital trends.
* **Relational Data Management:** Seamless integration with an SQLite database for structured and secure data querying.

## 🛠️ Tech Stack
* **Languages:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, statsmodels (ARIMA), Prophet
* **Web Framework:** Streamlit
* **Business Intelligence:** Power BI
* **Database:** SQLite
* **AI Integration:** Large Language Model (LLM) via API Key

## 🏗️ System Architecture & Data Flow
![System Architecture Diagram](images/DFD.png)



***
🚀 Installation & Setup

Follow these steps to run the CareCast Streamlit application on your local machine.

1. Clone the Repository
Bash
git clone [https://github.com/your-username/CareCast.git](https://github.com/your-username/CareCast.git)
cd CareCast
2. Create a Virtual Environment (Optional but Recommended)
Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure API Keys
To use the AI Chatbot feature, you need to provide your LLM API key.

Create a .env file in the root directory.

Add your API key like so:

Code snippet
LLM_API_KEY="your_api_key_here"
(Note: Ensure your .env file is added to .gitignore so your key is not exposed!)

5. Run the Application
Bash
streamlit run app.py
***

👥 Team
This project was collaboratively developed by:

Insha Farhan - Dashboard Architecture, Time-Series Modeling, Streamlit Deployment, AI Integration, Documentation.

Diksha Tiwari - Data Engineering, Preprocessing, Exploratory Data Analysis (EDA), Streamlit Development.

🔮 Future Scope
Deployment of a consumer-facing public portal for real-time bed availability tracking.

Integration with live Hospital Information System (HIS) APIs.

Advanced triage prediction using machine learning classification models.

If you find this project helpful, please consider giving it a ⭐!


***

### How to use this:
1. Create a new file in your GitHub repository named `README.md`.
2. Paste all the text inside the code block above into that file.
3. Update the `git clone` link with your actual GitHub username/URL.
4. Replace the placeholder image paths under the **Dashboards & Previews** section with the actual paths to your screenshots once you upload them to your repo.

URL of streamlit application:
https://healthcare-analytics-academic-project.streamlit.app/

Contributed by:
Insha Farhan and Diksha Tiwari



