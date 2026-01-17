# 🧠 Customer Micro-Trend Segmentation

An **AI-powered customer behavior analysis project** that goes beyond traditional segmentation to uncover **hidden micro-trends**, **rare customer behaviors**, and **actionable marketing insights** using unsupervised machine learning.

---

## 🚀 Project Overview

Customer Micro-Trend Segmentation analyzes raw transaction-level data and converts it into **meaningful behavioral segments**.  
Instead of only grouping customers broadly, this project identifies **fine-grained patterns** such as:

- Night shoppers
- Weekend-focused buyers
- Discount-sensitive customers
- High-value loyal customers
- Rare / anomalous buying behaviors

The final output is a **business-ready dashboard** with **clear marketing recommendations** for each customer cluster.

---

## ✨ Key Highlights

- End-to-end **data pipeline**
- Advanced **feature engineering**
- **K-Means + DBSCAN** clustering
- **PCA-based visualization**
- Multiple analytical graphs
- Automated CSV output
- **Sexy Streamlit dashboard**
- Clean, industry-level project structure

---

## 🧠 What Makes This Project Different?

| Traditional Projects    | This Project                   |
| ----------------------- | ------------------------------ |
| Broad segmentation only | Micro-trend discovery          |
| Only K-Means            | K-Means + DBSCAN               |
| Model-focused           | Business-action focused        |
| Static charts           | Interactive dashboard          |
| No recommendations      | Marketing strategy per cluster |

---

## 📊 Analytics & Visualizations

The project generates multiple insights such as:

- Purchases by Hour
- Weekend vs Weekday Purchases
- Order Value Distribution
- Elbow Method for K-Means
- PCA-based Cluster Visualization
- Revenue Contribution per Cluster
- Customer Behavior Profiles

---

## 🗂️ Project Structure

Customer_Micro_Trend_Segmentation/
│
├── data/
│ ├── raw/
│ │ └── transactions.csv
│ └── processed/
│ └── customer_micro_trend_segments.csv
│
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── clustering.py
│ ├── evaluation.py
│ └── visualization.py
│
├── scripts/
│ └── run_pipeline.py
│
├── dashboard/
│ └── dashboard.py
│
├── requirements.txt
└── README.md

---

## 🛠️ Tech Stack

- **Python 3.11**
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Plotly
- Streamlit

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Customer_Micro_Trend_Segmentation.git
cd Customer_Micro_Trend_Segmentation
```bash

2️⃣ Create & Activate Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Data Pipeline
python -m scripts.run_pipeline


➡ This generates:

data/processed/customer_micro_trend_segments.csv

5️⃣ Launch the Dashboard
streamlit run dashboard/dashboard.py

📁 Output File Description

customer_micro_trend_segments.csv contains:

Customer behavioral metrics

Cluster labels (K-Means & DBSCAN)

Marketing action suggestions

Example columns:

CustomerID
total_orders
avg_order_value
night_ratio
weekend_ratio
category_diversity
kmeans_cluster
dbscan_cluster
marketing_action

💡 Business Use Cases

Personalized marketing campaigns

Targeted discount strategies

Customer retention analysis

Identifying VIP and at-risk customers

Retail demand optimization

🔮 Future Enhancements

Real-time data ingestion

Model deployment on cloud

Advanced recommendation engine

Customer lifetime value (CLV) prediction

A/B testing for marketing strategies

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Saurabh Mishra
B.E. Computer Science Engineering
Passionate about AI, Data Science & Intelligent Systems

⭐ If you like this project, don’t forget to star the repository!
```
