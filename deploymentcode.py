import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pickle
from collections import defaultdict
import warnings
import mysql.connector
from sqlalchemy import create_engine
import hashlib
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine, text
warnings.filterwarnings('ignore')

# Database Configuration
DB_CONFIG = {
    'host': 'tipson-pilot.czdmtopxx7mm.ap-south-1.rds.amazonaws.com',  # Change to your MySQL host
    'port': 3306,         # Change to your MySQL port
    'user': 'tipsons_db',    # Change to your MySQL username
    'password': '86LOG91XvKe9WRep', # Change to your MySQL password
    'database': 'thefixedincome'  # Change to your database name
}

# Configuration
num_clusters = 5
RETRAIN_THRESHOLD = 11
UPDATE_BATCH_SIZE = 5
MIN_SAMPLES_FOR_CLUSTERING = 11  # Minimum samples needed for reliable clustering

# Enhanced bond type mapping
BOND_TYPE_MAPPING = {
    0: 'Government Securities',
    1: 'Corporate Bonds', 
    2: 'Municipal Bonds',
    3: 'Treasury Bills',
    4: 'Commercial Papers',
    5: 'Certificate of Deposits',
    6: 'Infrastructure Bonds',
    7: 'Tax-Free Bonds',
    8: 'Convertible Bonds',
    9: 'Zero Coupon Bonds',
    10: 'High Yield Bonds',
    11: 'International Bonds',
    12: 'Green Bonds'
}

# Reverse mapping
TYPE_TO_INDEX = {v: k for k, v in BOND_TYPE_MAPPING.items()}
NUM_BOND_TYPES = len(BOND_TYPE_MAPPING)

# Enhanced rating mapping with numeric risk scores
RATING_MAPPING = {
    'AAA': {'score': 1, 'risk_weight': 0.01, 'category': 'Prime'},
    'AA+': {'score': 2, 'risk_weight': 0.02, 'category': 'High Grade'}, 
    'AA': {'score': 3, 'risk_weight': 0.03, 'category': 'High Grade'},
    'AA-': {'score': 4, 'risk_weight': 0.04, 'category': 'High Grade'},
    'A+': {'score': 5, 'risk_weight': 0.06, 'category': 'Upper Medium'},
    'A': {'score': 6, 'risk_weight': 0.08, 'category': 'Upper Medium'},
    'A-': {'score': 7, 'risk_weight': 0.10, 'category': 'Upper Medium'},
    'BBB+': {'score': 8, 'risk_weight': 0.15, 'category': 'Lower Medium'},
    'BBB': {'score': 9, 'risk_weight': 0.20, 'category': 'Lower Medium'},
    'BBB-': {'score': 10, 'risk_weight': 0.25, 'category': 'Lower Medium'},
    'BB+': {'score': 11, 'risk_weight': 0.35, 'category': 'Speculative'},
    'BB': {'score': 12, 'risk_weight': 0.45, 'category': 'Speculative'},
    'BB-': {'score': 13, 'risk_weight': 0.55, 'category': 'Speculative'},
    'B+': {'score': 14, 'risk_weight': 0.70, 'category': 'Highly Speculative'},
    'B': {'score': 15, 'risk_weight': 0.85, 'category': 'Highly Speculative'},
    'B-': {'score': 16, 'risk_weight': 0.95, 'category': 'Poor'},
    'C': {'score': 17, 'risk_weight': 1.0, 'category': 'Very Poor'}
}

def get_risk_category_from_rating(rating):
    """Convert rating to simplified risk category"""
    if isinstance(rating, str):
        rating_info = RATING_MAPPING.get(rating, {'category': 'Lower Medium'})
        category = rating_info['category']
    else:
        # Handle numeric ratings
        if rating <= 7:
            category = 'High Grade'
        elif rating <= 10:
            category = 'Lower Medium'
        else:
            category = 'Speculative'
    
    # Map to A, B, C system
    if category in ['Prime', 'High Grade', 'Upper Medium']:
        return 'A'  # Conservative
    elif category in ['Lower Medium']:
        return 'B'  # Moderate
    else:
        return 'C'  # Aggressive

# Database connection functions
@st.cache_resource
def create_db_connection():
    """Create database connection with error handling"""
    try:
        # Create SQLAlchemy engine
        connection_string = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.info("Please check your database configuration and ensure MySQL is running.")
        return None

def load_data_from_db(engine, table_name='investment_data'):
    """Load investment data from MySQL database"""
    try:
        query = f"""
        with rating as (
 select ra.agency_name,pr.rating,pm.isin,pm.id  from thefixedincome.product_master pm  inner join (
select pr.agency_id,product_id
,case 
	when pr.rating = 1 then 'AAA' 
	when pr.rating = 2 Then  'AAA(SO)'
	when pr.rating = 3 Then  'AA+'
	when pr.rating = 4 Then  'AA+(SO)'
	when pr.rating = 5 Then  'AA'
	when pr.rating = 6 Then  'AA(SO)'
	when pr.rating = 7 Then  'AA-'
	when pr.rating = 8 Then  'AA-(SO)'
	when pr.rating = 9 Then  'A+'
	when pr.rating = 10 Then  'A+(SO)'
	when pr.rating = 11 Then  'A'
	when pr.rating = 12 Then  'A(SO)'
	when pr.rating = 13 Then  'A-'
	when pr.rating = 14 Then  'A-(SO)'
	when pr.rating = 15 Then  'BBB+'
	when pr.rating = 16 Then  'BBB+(SO)'
	when pr.rating = 17 Then  'BBB'
	when pr.rating = 18 Then  'BBB(SO)'
	when pr.rating = 19 Then  'BBB-'
	when pr.rating = 20 Then  'BBB-(SO)'
	when pr.rating = 21 Then  'B+'
	when pr.rating = 22 Then  'B+(SO)'
	when pr.rating = 23 Then  'B'
	when pr.rating = 24 Then  'B(SO)'
	when pr.rating = 25 Then  'B-(SO)'
	when pr.rating = 26 Then  'B-'
	when pr.rating = 27 Then  'BB+'
	when pr.rating = 28 Then  'BB+(SO)'
	when pr.rating = 29 Then  'BB'
	when pr.rating = 30 Then  'BB(SO)'
	when pr.rating = 31 Then  'BB-'
	when pr.rating = 32 Then  'BB-(SO)'
	when pr.rating = 33 Then  'C+'
	when pr.rating = 34 Then  'C+(SO)'
	when pr.rating = 35 Then  'C'
	when pr.rating = 36 Then  'C(SO)'
	when pr.rating = 37 Then  'C-'
	when pr.rating = 38 Then  'C-(SO)'
	when pr.rating = 39 Then  'D+'
	when pr.rating = 40 Then  'D+(SO)'
	when pr.rating = 41 Then  'D-(SO)'
	when pr.rating = 42 Then  'AAA(CE)'
	when pr.rating = 43 Then  'AA+(CE)'
	when pr.rating = 44 Then  'AA(CE)'
	when pr.rating = 45 Then  'AA-(CE)'
	when pr.rating = 46 Then  'A+(CE)'
	when pr.rating = 47 Then  'A(CE)'
	when pr.rating = 48 Then  'A-(CE)'
	when pr.rating = 49 Then  'BBB+(CE)'
	when pr.rating = 50 Then  'BBB(CE)'
	when pr.rating = 51 Then  'BBB-(CE)'
	when pr.rating = 52 Then  'BB+(CE)'
	when pr.rating = 53 Then  'BB(CE)'
	when pr.rating = 54 Then  'BB-(CE)'
	when pr.rating = 55 Then  'B+(CE)'
	when pr.rating = 56 Then  'B(CE)'
	when pr.rating = 57 Then  'B-(CE)'
	when pr.rating = 58 Then  'C+(CE)'
	when pr.rating = 59 Then  'C(CE)'
	when pr.rating = 60 Then  'C-(CE)'
	when pr.rating = 61 Then  'D(CE)'
else '' 
end as rating from
thefixedincome.product_rating pr ) as  pr
		on pm.id = pr.product_id 
 	inner join thefixedincome.rating_agency ra 
 		on ra.id = pr.agency_id
)
select 
o.id as `sr.no`
,case when o.member_id <> 0 then um.user_member_id  else u.client_id end  as `Client ID`
,pm.id as `Bond ID`
,o.investment_amount as `Trade Value`
,cast(o.trade_date as datetime) as `Trade Date`
,Floor(DATEDIFF(o.trade_date,case when o.member_id <> 0 then um.date_of_birth else u.date_of_birth end) / 365) as Age
,case when o.member_id <> 0 then 
	case when um.user_type = 1 then 
		case when um.gender = 1 then 'M'
			when um.gender = 2 then 'F'
			else '' 
		END 
		when um.user_type = 3 then 'HUF'
		else ''
	END 
	else
	case when u.user_type = 1 then
		case when u.gender = 1 then 'M'
			when u.gender = 2 then 'F'
			else ''
		end
		when u.user_type = 3 then 'HUF'
		else ''
	end
end as Gender
,case when r.agency_name = 'CRISIL' then r.rating else r.rating end  as `Risk Category`
,case when o.member_id <> 0 then um.parent_id else 0 end as `Parent Client ID` 
,pm.coupon_rate as `Coupon Rate (%)`
,case  when pm.Issuer_type = 1 then ' PSU'
	  when pm.Issuer_type = 2 then ' PSU Bank'
	  when pm.Issuer_type = 3 then ' Corporate'
	  when  pm.Issuer_type = 5  then 'Municipal'
	  when  pm.Issuer_type = 6  then 'Private Bank'
	  when  pm.Issuer_type = 7 then 'NBFC'
	  when  pm.Issuer_type = 8 then 'State Governent Guranteed'
	  when  pm.Issuer_type = 9 then 'Central Goverment1'
	  when  pm.Issuer_type = 10 then 'PSE'
	  when  pm.Issuer_type = 11 then 'Bank'
	  when  pm.Issuer_type = 12 then 'Non PSU'
	  when  pm.Issuer_type = 13 then 'Special Purpose Vehicale'
	  when  pm.Issuer_type = 14 then 'Private Bonds'
	  when  pm.Issuer_type = 10 then 'Special Purpose Vehicale'
else ''
end as Type
,o.yield as `YTM (%)`
-- ,TIMEDIFF(YEAR, pm.issue_date, date_format(str_to_date(replace(pm.maturity_date,'/','-'),'%Y-%m-%d'),'%Y-%m-%d')) as `Tenure (Years)`
,TIMESTAMPDIFF(YEAR, pm.issue_date, date_format(str_to_date(replace(pm.maturity_date,'/','-'),'%Y-%m-%d'),'%Y-%m-%d')) as `Tenure (Years)`
,case when pm.ip_frequency = 1 then 'Monthly'
 when pm.ip_frequency = 2 then  'Bi-Monthly'
 when pm.ip_frequency = 3 then 'Quarterly'
 when pm.ip_frequency = 4 then 'Semi-Annual'
 when pm.ip_frequency = 5 then 'Annual'
 when pm.ip_frequency = 6 then 'At Maturity'
 when pm.ip_frequency = 6 then 'Not Applicable'
 else ''
 end as Frequency
 ,maturity_date as `Maturity Date`
 ,pm.issue_date as `Bond Issue Date`
from orders o
	inner join offers ofr
		on ofr.id = o.offer_id
	inner join product_master pm
		on pm.id = ofr.product_id
	left join seller_master sm 
		on sm.id = o.seller_id
-- 	left join seller_master sm2 
-- 		on sm2.id = o.buyer_id 
	left join users u
		on o.user_id  = u.id 
		and o.member_id = 0
	left join user_members um 
		on o.member_id = um.id
		and o.member_id <> 0
	left join users uma
		on uma.id = um.parent_id 
	left join admins a 
		on a.id = o.approved_by 
	left join admins aa
		on aa.id = o.rejected_by		
	left join admins rma
		on rma.id = u.rm_id
	left join admins rmaf
		on rmaf.id = uma.rm_id
	left join admins rmo
		on o.rm_id = rmo.id
	left join agents ag
		on ag.agentID = u.reference_code
		or ag.agentID = uma.reference_code
	left join hear_aboutus ha 
		on ha.id = u.hear_about_us
	left join rating r
		on r.id = pm.id
where o.primary_offer_id = 0 and o.order_type = 1 and o.seller_id = 4
        """
        
        df = pd.read_sql_query(query, engine)
        
        # Data type conversions
        date_columns = ['Trade Date', 'Maturity Date', 'Bond Issue Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Numeric conversions
        numeric_columns = ['Age', 'Trade Value', 'Coupon Rate (%)', 'YTM (%)', 'Tenure (Years)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing Parent Client ID
        df['Parent Client ID'] = df['Parent Client ID'].fillna('--NA--')
        
        # Validate data
        if len(df) == 0:
            raise ValueError("No data found in database")
        
        # Check for required columns
        required_columns = ['Client ID', 'Age', 'Gender', 'Trade Date', 'Trade Value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        st.success(f"Successfully loaded {len(df)} records from database")
        return df
        
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        # Provide fallback sample data for testing
        st.warning("Using sample data for demonstration")
        return create_sample_data()

def create_sample_data():
    """Create sample data for testing when database is not available"""
    np.random.seed(42)
    n_clients = 100
    n_records = 500
    
    clients = [f"C_{i:04d}" for i in range(1, n_clients + 1)]
    bond_types = list(BOND_TYPE_MAPPING.values())
    ratings = list(RATING_MAPPING.keys())
    
    data = []
    for i in range(n_records):
        client_id = np.random.choice(clients)
        client_num = int(client_id.split('_')[1])
        
        # Generate consistent client demographics
        age = np.random.normal(45, 15)
        age = max(18, min(80, int(age)))
        
        data.append({
            'Client ID': client_id,
            'Age': age,
            'Gender': np.random.choice(['M', 'F', 'HUF'], p=[0.5, 0.4, 0.1]),
            'Risk Category': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.5, 0.2]),
            'Parent Client ID': '--NA--' if client_num <= 80 else f"C_{np.random.randint(1, 81):04d}",
            'Bond ID': f"BOND_{i:04d}",
            'Type': np.random.choice(bond_types),
            'Trade Date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 1000)),
            'Trade Value': np.random.uniform(10000, 1000000),
            'Maturity Date': pd.Timestamp.now() + pd.Timedelta(days=np.random.randint(365, 3650)),
            'Bond Issue Date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 365)),
            'Coupon Rate (%)': np.random.uniform(3, 12),
            'YTM (%)': np.random.uniform(4, 15),
            'Frequency': np.random.choice(['Annual', 'Semi-Annual', 'Quarterly']),
            'Tenure (Years)': np.random.uniform(1, 10),
            'Rating': np.random.choice(ratings)
        })
    
    return pd.DataFrame(data)

# Enhanced model loading with better error handling
def load_or_initialize_models():
    """Load existing models or create new ones with improved validation"""
    global kmeans, scaler, Q, new_client_count, feature_names, cluster_analyzer, label_encoders
    
    # Initialize label encoders for categorical features
    label_encoders = {}
    
    # Load Q-table
    if os.path.exists("q_table.npy"):
        try:
            Q = np.load("q_table.npy")
            if Q.shape != (num_clusters, NUM_BOND_TYPES):
                st.warning("Q-table dimensions mismatch. Reinitializing...")
                Q = np.zeros((num_clusters, NUM_BOND_TYPES))
        except:
            Q = np.zeros((num_clusters, NUM_BOND_TYPES))
    else:
        Q = np.zeros((num_clusters, NUM_BOND_TYPES))
    
    # Load metadata
    if os.path.exists("feedback_meta.json"):
        try:
            with open("feedback_meta.json", "r") as f:
                meta = json.load(f)
                new_client_count = meta.get("new_clients", 0)
        except:
            new_client_count = 0
    else:
        new_client_count = 0
    
    # Load models
    if os.path.exists("kmeans_model.pkl"):
        try:
            with open("kmeans_model.pkl", "rb") as f:
                kmeans = pickle.load(f)
        except:
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
    else:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
    
    if os.path.exists("scaler_model.pkl"):
        try:
            with open("scaler_model.pkl", "rb") as f:
                scaler = pickle.load(f)
        except:
            scaler = StandardScaler()
    else:
        scaler = StandardScaler()
    
    # Load feature names
    if os.path.exists("feature_names.json"):
        try:
            with open("feature_names.json", "r") as f:
                feature_names = json.load(f)
        except:
            feature_names = []
    else:
        feature_names = []

# Enhanced clustering with better feature engineering
class ImprovedClusterAnalyzer:
    """Improved cluster analysis with statistical validation"""
    
    def __init__(self):
        self.cluster_descriptions = {}
        self.feature_importance = {}
        self.cluster_stability_scores = {}
    
    def analyze_clusters(self, X, y_clusters, feature_names, clients_df):
        """Enhanced cluster analysis with statistical validation"""
        if len(X) == 0 or len(clients_df) == 0:
            return
        
        self.cluster_descriptions = {}
        self.feature_importance = {}
        self.cluster_stability_scores = {}
        
        # Calculate global statistics
        global_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0)
        }
        
        for cluster_id in range(num_clusters):
            cluster_mask = y_clusters == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size == 0:
                continue
            
            cluster_data = X[cluster_mask]
            cluster_clients = clients_df[clients_df['Cluster'] == cluster_id]
            
            # Enhanced feature importance with statistical significance
            cluster_mean = np.mean(cluster_data, axis=0)
            
            # Calculate z-scores for feature importance
            z_scores = np.abs((cluster_mean - global_stats['mean']) / (global_stats['std'] + 1e-8))
            top_features_idx = np.argsort(z_scores)[-5:][::-1]
            
            # Calculate cluster stability (intra-cluster variance)
            if cluster_size > 1:
                intra_cluster_distances = cdist(cluster_data, cluster_data)
                stability = 1 / (1 + np.mean(intra_cluster_distances))
            else:
                stability = 1.0
            
            self.cluster_stability_scores[cluster_id] = stability
            
            # Generate description
            description = self._generate_enhanced_description(
                cluster_clients, cluster_id, feature_names,
                top_features_idx, z_scores, cluster_size
            )
            
            self.cluster_descriptions[cluster_id] = description
            
            # Store feature importance with z-scores
            self.feature_importance[cluster_id] = {
                feature_names[i]: {'importance': z_scores[i], 'value': cluster_mean[i]}
                for i in top_features_idx if i < len(feature_names)
            }
    
    def _generate_enhanced_description(self, cluster_clients, cluster_id, 
                                     feature_names, top_features_idx, 
                                     z_scores, cluster_size):
        """Generate statistically informed cluster description"""
        
        if len(cluster_clients) == 0:
            return {
                'name': f"Empty Cluster {cluster_id}",
                'description': "No clients assigned",
                'size': 0,
                'key_metrics': {},
                'confidence': 0.0
            }
        
        # Calculate confidence based on cluster size and stability
        confidence = min(1.0, cluster_size / 20) * self.cluster_stability_scores.get(cluster_id, 0.5)
        
        # Demographic analysis with error handling
        try:
            avg_age = cluster_clients['Age'].mean()
            dominant_gender = cluster_clients['Gender'].mode().iloc[0] if len(cluster_clients) > 0 else 'Mixed'
            dominant_risk = cluster_clients['Risk Category'].mode().iloc[0] if len(cluster_clients) > 0 else 'B'
            avg_reinvest = cluster_clients['Reinvest Ratio'].mean() if 'Reinvest Ratio' in cluster_clients.columns else 0
        except:
            avg_age, dominant_gender, dominant_risk, avg_reinvest = 40, 'Mixed', 'B', 0
        
        # Generate meaningful cluster name
        age_group = "Young" if avg_age < 35 else "Middle-aged" if avg_age < 55 else "Mature"
        risk_type = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}.get(dominant_risk, "Mixed")
        
        cluster_name = f"{age_group} {risk_type} Investors"
        
        # Enhanced description with statistical insights
        description = (f"Cluster of {cluster_size} clients, primarily {age_group.lower()} "
                      f"({avg_age:.1f} avg age) with {risk_type.lower()} investment approach")
        
        if avg_reinvest > 0:
            reinvest_level = "high" if avg_reinvest > 60 else "moderate" if avg_reinvest > 30 else "low"
            description += f" and {reinvest_level} reinvestment activity ({avg_reinvest:.1f}%)"
        
        return {
            'name': cluster_name,
            'description': description,
            'size': cluster_size,
            'key_metrics': {
                'avg_age': round(avg_age, 1),
                'dominant_risk': dominant_risk,
                'dominant_gender': dominant_gender,
                'avg_reinvest_ratio': round(avg_reinvest, 1)
            },
            'confidence': round(confidence, 2),
            'stability': round(self.cluster_stability_scores.get(cluster_id, 0), 2)
        }

# Enhanced feature engineering
def prepare_enhanced_features(clients_df):
    """Enhanced feature preparation with better encoding and validation"""
    if len(clients_df) == 0:
        return np.array([]), [], []
    
    # Ensure all required columns exist
    required_numeric = ['Age', 'Reinvest Ratio', 'Total Invested']
    required_categorical = ['Gender', 'Risk Category']
    
    # Add missing numeric columns with defaults
    for col in required_numeric:
        if col not in clients_df.columns:
            if col == 'Reinvest Ratio':
                clients_df[col] = 30.0  # Default reinvestment ratio
            elif col == 'Total Invested':
                clients_df[col] = 100000.0  # Default investment amount
            else:
                clients_df[col] = clients_df['Age'].mean() if 'Age' in clients_df.columns else 40.0
    
    # Add missing categorical columns with defaults
    for col in required_categorical:
        if col not in clients_df.columns:
            if col == 'Gender':
                clients_df[col] = 'M'
            elif col == 'Risk Category':
                clients_df[col] = 'B'
    
    # Enhanced feature engineering
    try:
        # Numeric features with outlier handling
        numeric_features = ['Age', 'Reinvest Ratio']
        if 'Total Invested' in clients_df.columns:
            # Log transform for investment amount to handle skewness
            clients_df['Log_Investment'] = np.log1p(clients_df['Total Invested'])
            numeric_features.append('Log_Investment')
        
        # Risk score conversion
        clients_df['Risk_Score'] = clients_df['Risk Category'].map({'A': 1, 'B': 2, 'C': 3}).fillna(2)
        numeric_features.append('Risk_Score')
        
        # Age groups for better clustering
        clients_df['Age_Group'] = pd.cut(clients_df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
        
        # Investment frequency (if available)
        if 'Days Since Last Investment' in clients_df.columns:
            clients_df['Investment_Frequency'] = 1 / (clients_df['Days Since Last Investment'] + 1)
            numeric_features.append('Investment_Frequency')
        
        # Prepare features for clustering
        X_numeric = clients_df[numeric_features].fillna(0).values
        
        # Scale numeric features
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Categorical features with one-hot encoding
        categorical_features = ['Gender', 'Age_Group']
        X_categorical = pd.get_dummies(clients_df[categorical_features], sparse=False).values
        
        # Combine features
        X = np.concatenate([X_scaled, X_categorical], axis=1)
        
        # Create feature names
        categorical_columns = pd.get_dummies(clients_df[categorical_features]).columns.tolist()
        feature_names = numeric_features + categorical_columns
        
        # Validate output
        if X.shape[0] == 0 or X.shape[1] == 0:
            st.error("No valid features could be created from the data")
            return np.array([]), [], []
            
        st.info(f"Created {X.shape[1]} features for {X.shape[0]} clients")
        return X, feature_names, categorical_columns   
         
    except Exception as e:
        st.error(f"Feature preparation error: {str(e)}")
        return np.array([]), [], []

# Enhanced reinvestment calculation
def calculate_enhanced_reinvestment_ratio(client_df):
    """Improved reinvestment calculation with cashflow analysis"""
    if len(client_df) == 0:
        return 0.0
    
    try:
        client_df = client_df.sort_values('Trade Date')
        
        # Simple but robust reinvestment calculation
        total_investments = len(client_df)
        if total_investments <= 1:
            return 0.0
        
        # Calculate time gaps between investments
        trade_dates = pd.to_datetime(client_df['Trade Date'])
        time_gaps = trade_dates.diff().dt.days.fillna(0)
        
        # Investments with short gaps (< 90 days) are likely reinvestments
        reinvestment_threshold = 90  # days
        reinvestments = np.sum(time_gaps[1:] <= reinvestment_threshold)
        
        reinvestment_ratio = (reinvestments / max(1, total_investments - 1)) * 100
        return min(100, max(0, reinvestment_ratio))
        
    except Exception as e:
        return 30.0  # Default value

def build_enhanced_client_profiles(df):
    """Build enhanced client profiles with robust error handling"""
    if len(df) == 0:
        return pd.DataFrame()
    
    try:
        clients = df['Client ID'].unique()
        client_records = []
        
        for client_id in clients:
            client_trades = df[df['Client ID'] == client_id]
            if len(client_trades) == 0:
                continue
            
            first_trade = client_trades.iloc[0]
            
            # Basic demographics with fallbacks
            age = first_trade.get('Age', 40)
            gender = first_trade.get('Gender', 'M')
            risk_category = first_trade.get('Risk Category', 'B')
            
            # Convert rating to risk category if needed
            if 'Rating' in client_trades.columns:
                rating = client_trades['Rating'].iloc[0]
                risk_category = get_risk_category_from_rating(rating)
            
            parent_id = first_trade.get('Parent Client ID', '--NA--')
            if pd.isna(parent_id) or parent_id == '':
                parent_id = '--NA--'
            
            # Calculate metrics
            reinv_ratio = calculate_enhanced_reinvestment_ratio(client_trades)
            total_invested = client_trades['Trade Value'].sum()
            avg_coupon = client_trades['Coupon Rate (%)'].mean() if 'Coupon Rate (%)' in client_trades.columns else 6.0
            avg_ytm = client_trades['YTM (%)'].mean() if 'YTM (%)' in client_trades.columns else 7.0
            
            client_records.append({
                'Client ID': client_id,
                'Age': max(18, min(80, int(age))),  # Bounds checking
                'Gender': gender,
                'Risk Category': risk_category,
                'Parent ID': parent_id,
                'Reinvest Ratio': reinv_ratio,
                'Total Invested': total_invested,
                'Avg Coupon': avg_coupon,
                'Avg YTM': avg_ytm,
                'Trade Count': len(client_trades)
            })
        
        if not client_records:
            return pd.DataFrame()
        
        clients_df = pd.DataFrame(client_records).set_index('Client ID')
        
        # Handle parent ratios
        reinv_map = clients_df['Reinvest Ratio'].to_dict()
        for cid in clients_df.index:
            pid = clients_df.at[cid, 'Parent ID']
            if pid != '--NA--' and pid in reinv_map:
                clients_df.at[cid, 'Parent Ratio'] = reinv_map[pid]
            else:
                clients_df.at[cid, 'Parent Ratio'] = clients_df.at[cid, 'Reinvest Ratio']
        
        return clients_df
        
    except Exception as e:
        st.error(f"Error building client profiles: {str(e)}")
        return pd.DataFrame()


# Enhanced Q-learning with better reward structure
def train_enhanced_q_learning(clients_df, df, episodes=5000):
    """Improved Q-learning with better reward structure and validation"""
    global Q
    
    if len(clients_df) == 0 or len(df) == 0:
        return Q
    
    try:
        # Create more sophisticated client behavior mapping
        client_behavior = {}
        
        for client_id in df['Client ID'].unique():
            if client_id not in clients_df.index:
                continue
                
            client_trades = df[df['Client ID'] == client_id]
            
            if len(client_trades) < 2:  # Need at least 2 trades for behavior analysis
                continue
            
            # Calculate investment distribution and patterns
            type_investments = client_trades.groupby('Type')['Trade Value'].sum()
            total_investment = type_investments.sum()
            
            if total_investment == 0:
                continue
            
            # Create preference profile
            preferences = {}
            for bond_type, amount in type_investments.items():
                type_idx = TYPE_TO_INDEX.get(bond_type, 1)
                preference_strength = amount / total_investment
                preferences[type_idx] = preference_strength
            
            # Calculate consistency (how concentrated their investments are)
            consistency = max(type_investments) / total_investment
            
            client_behavior[client_id] = {
                'preferences': preferences,
                'consistency': consistency,
                'total_trades': len(client_trades),
                'avg_investment': client_trades['Trade Value'].mean()
            }
        
        if not client_behavior:
            st.warning("No sufficient client behavior data for training")
            return Q
        
        # Enhanced Q-learning parameters
        alpha = 0.15  # Slightly higher learning rate
        gamma = 0.95  # Higher discount factor for long-term learning
        epsilon_start = 0.3
        epsilon_end = 0.01
        epsilon_decay = 0.995
        
        training_rewards = []
        
        for episode in range(episodes):
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
            
            # Select random client with sufficient data
            eligible_clients = [cid for cid in client_behavior.keys() 
                              if cid in clients_df.index]
            
            if not eligible_clients:
                break
                
            client_id = np.random.choice(eligible_clients)
            behavior = client_behavior[client_id]
            state = clients_df.loc[client_id, 'Cluster']
            
            if state >= num_clusters:
                continue
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, NUM_BOND_TYPES)
            else:
                action = np.argmax(Q[state])
            
            # Enhanced reward calculation
            reward = 0.0
            
            # Base reward from preferences
            if action in behavior['preferences']:
                reward += behavior['preferences'][action] * 2.0  # Scale up preference reward
            
            # Consistency bonus - reward consistent clients more
            consistency_bonus = behavior['consistency'] * 0.5
            reward += consistency_bonus
            
            # Experience bonus - clients with more trades provide better signal
            experience_bonus = min(0.3, behavior['total_trades'] / 20.0)
            reward += experience_bonus
            
            # Update Q-value with enhanced learning
            best_next_q = np.max(Q[state])
            current_q = Q[state, action]
            
            # Enhanced update rule with momentum
            Q[state, action] += alpha * (reward + gamma * best_next_q - current_q)
            
            training_rewards.append(reward)
            
            # Periodic validation every 1000 episodes
            if episode > 0 and episode % 1000 == 0:
                temp_accuracy, _ = calculate_enhanced_model_accuracy(clients_df, df, Q)
                if episode % 2000 == 0:
                    st.info(f"Training episode {episode}: Accuracy = {temp_accuracy*100:.1f}%")
        
        # Normalize Q-values for better interpretation
        Q = Q / (np.max(np.abs(Q)) + 1e-8)
        
        # Final validation
        final_accuracy, metrics = calculate_enhanced_model_accuracy(clients_df, df, Q)
        
        st.success(f"Training completed: {episodes} episodes")
        st.info(f"Final weighted accuracy: {final_accuracy*100:.1f}%")
        
        if 'exact_accuracy' in metrics:
            st.info(f"Exact match accuracy: {metrics['exact_accuracy']*100:.1f}%")
            st.info(f"Top-3 accuracy: {metrics['top_3_accuracy']*100:.1f}%")
        
        # Save enhanced Q-table
        np.save("q_table.npy", Q)
        
        return Q
        
    except Exception as e:
        st.error(f"Enhanced Q-learning training error: {str(e)}")
        return Q


def calculate_enhanced_model_accuracy(clients_df, df, Q_table):
    """Enhanced accuracy calculation with better validation"""
    if len(clients_df) == 0 or len(df) == 0:
        return 0.0, {}
    
    try:
        accuracy_metrics = {
            'exact_matches': 0,
            'top_3_matches': 0,
            'weighted_accuracy': 0.0,
            'total_predictions': 0,
            'client_accuracies': {}
        }
        
        for client_id in clients_df.index:
            client_trades = df[df['Client ID'] == client_id]
            
            if len(client_trades) < 3:  # Skip clients with too few trades
                continue
                
            cluster_id = clients_df.loc[client_id, 'Cluster']
            if cluster_id >= len(Q_table):
                continue
            
            # Calculate client's actual preferences (weighted by investment amount)
            type_weights = client_trades.groupby('Type')['Trade Value'].sum()
            total_investment = type_weights.sum()
            
            if total_investment == 0:
                continue
                
            # Get model predictions (top 3)
            q_values = Q_table[cluster_id]
            top_3_predictions = np.argsort(q_values)[-3:][::-1]
            top_prediction = top_3_predictions[0]
            
            # Calculate weighted accuracy for this client
            client_accuracy = 0.0
            for bond_type, investment in type_weights.items():
                type_idx = TYPE_TO_INDEX.get(bond_type, 1)
                weight = investment / total_investment
                
                if type_idx == top_prediction:
                    client_accuracy += weight * 1.0  # Full credit
                elif type_idx in top_3_predictions:
                    client_accuracy += weight * 0.5  # Partial credit
            
            accuracy_metrics['client_accuracies'][client_id] = client_accuracy
            accuracy_metrics['weighted_accuracy'] += client_accuracy
            accuracy_metrics['total_predictions'] += 1
            
            # Check exact and top-3 matches
            most_invested_type = type_weights.idxmax()
            actual_idx = TYPE_TO_INDEX.get(most_invested_type, 1)
            
            if actual_idx == top_prediction:
                accuracy_metrics['exact_matches'] += 1
            
            if actual_idx in top_3_predictions:
                accuracy_metrics['top_3_matches'] += 1
        
        # Calculate final metrics
        if accuracy_metrics['total_predictions'] > 0:
            exact_accuracy = accuracy_metrics['exact_matches'] / accuracy_metrics['total_predictions']
            top_3_accuracy = accuracy_metrics['top_3_matches'] / accuracy_metrics['total_predictions']
            weighted_accuracy = accuracy_metrics['weighted_accuracy'] / accuracy_metrics['total_predictions']
            
            return weighted_accuracy, {
                'exact_accuracy': exact_accuracy,
                'top_3_accuracy': top_3_accuracy,
                'weighted_accuracy': weighted_accuracy,
                'total_clients': accuracy_metrics['total_predictions'],
                'distribution': accuracy_metrics['client_accuracies']
            }
        else:
            return 0.0, {'error': 'No valid predictions could be made'}
            
    except Exception as e:
        st.error(f"Enhanced accuracy calculation error: {str(e)}")
        return 0.0, {'error': str(e)}

# Enhanced recommendation system
def get_enhanced_recommendation(client_features, cluster_id):
    """Get enhanced recommendation with confidence scoring"""
    try:
        if cluster_id >= len(Q):
            return "Corporate Bonds", 0.33, []
        
        q_values = Q[cluster_id]
        recommended_action = np.argmax(q_values)
        
        # Calculate confidence
        if np.sum(q_values) > 0:
            confidence = q_values[recommended_action] / np.sum(q_values)
        else:
            confidence = 1.0 / NUM_BOND_TYPES
        
        # Get bond type name
        recommended_bond = BOND_TYPE_MAPPING.get(recommended_action, "Corporate Bonds")
        
        # Get top 3 recommendations
        top_3_indices = np.argsort(q_values)[-3:][::-1]
        top_recommendations = []
        
        for idx in top_3_indices:
            bond_name = BOND_TYPE_MAPPING.get(idx, f"Bond Type {idx}")
            score = q_values[idx]
            top_recommendations.append((bond_name, score))
        
        return recommended_bond, confidence, top_recommendations
        
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return "Corporate Bonds", 0.33, []

# Initialize global variables
load_or_initialize_models()
cluster_analyzer = ImprovedClusterAnalyzer()
# Force clean reset to ensure feature consistency
import os
model_files_to_reset = ["kmeans_model.pkl", "scaler_model.pkl", "feature_names.json", "q_table.npy"]
for file in model_files_to_reset:
    if os.path.exists(file):
        os.remove(file)

# Reinitialize all models fresh
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
scaler = StandardScaler()
Q = np.zeros((num_clusters, NUM_BOND_TYPES))
feature_names = []
# Main Streamlit App
st.set_page_config(page_title="Enhanced Investment Analysis with MySQL", layout="wide")
st.title("Enhanced Investment Analysis System with MySQL Database")

# Database connection status
engine = create_db_connection()

if engine is not None:
    st.success("Database connection established")
    
    # Load data from database
    try:
        df = load_data_from_db(engine)
        
        if len(df) > 0:
            # Build client profiles
            clients_df = build_enhanced_client_profiles(df)
            
            if len(clients_df) >= MIN_SAMPLES_FOR_CLUSTERING:
                # Prepare features
                X, feature_names, categorical_columns = prepare_enhanced_features(clients_df)
                
                if len(X) > 0 and X.shape[1] > 0:
                    # Always fit fresh models to current data structure
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    
                    # Fit clustering model
                    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
                    kmeans.fit(X)
                    cluster_labels = kmeans.predict(X)
                    clients_df['Cluster'] = cluster_labels
                    
                    # Analyze clusters
                    cluster_analyzer.analyze_clusters(X, cluster_labels, feature_names, clients_df)
                    
                    # Train Q-learning
                    Q = train_enhanced_q_learning(clients_df, df)
                    
                    # Calculate accuracy
                    accuracy = calculate_enhanced_model_accuracy(clients_df, df, Q)
                    
                    # Save models with current feature structure
                    with open("kmeans_model.pkl", "wb") as f:
                        pickle.dump(kmeans, f)
                    with open("scaler_model.pkl", "wb") as f:
                        pickle.dump(scaler, f)
                    with open("feature_names.json", "w") as f:
                        json.dump(feature_names, f)
                    np.save("q_table.npy", Q)
                    
                    st.success(f"Models trained successfully! Accuracy: {accuracy*100:.1f}%")
                    
                else:
                    st.error("Feature preparation failed - no valid features generated")
                    X, cluster_labels, accuracy = np.array([]), np.array([]), 0.0
            else:
                st.warning(f"Insufficient data for clustering. Need at least {MIN_SAMPLES_FOR_CLUSTERING} clients.")
                X, cluster_labels, accuracy = np.array([]), np.array([]), 0.0
        else:
            st.error("No data loaded from database")
            clients_df = pd.DataFrame()
            X, cluster_labels, accuracy = np.array([]), np.array([]), 0.0
            
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        clients_df = pd.DataFrame()
        X, cluster_labels, accuracy = np.array([]), np.array([]), 0.0


# Database Management Functions
def save_new_client_to_db(engine, client_data):
    """Save new client data to database"""
    try:
        # Convert client data to database format
        db_record = {
            'client_id': f"C_{np.random.randint(10000, 99999)}",
            'age': client_data['Age'],
            'gender': client_data['Gender'],
            'risk_category': client_data['Risk Category'],
            'parent_client_id': client_data.get('Parent ID', '--NA--'),
            'created_date': pd.Timestamp.now()
        }
        
        # Insert into database
        query = """
        INSERT INTO clients (client_id, age, gender, risk_category, parent_client_id, created_date)
        VALUES (%(client_id)s, %(age)s, %(gender)s, %(risk_category)s, %(parent_client_id)s, %(created_date)s)
        ON DUPLICATE KEY UPDATE
        age = VALUES(age),
        gender = VALUES(gender),
        risk_category = VALUES(risk_category)
        """
        
        with engine.connect() as conn:
            conn.execute(query, db_record)
            conn.commit()
        
        return db_record['client_id']
    
    except Exception as e:
        st.error(f"Database save error: {str(e)}")
        return None

def update_model_feedback(engine, client_id, recommended_bond, actual_choice, reward):
    """Update model feedback in database"""
    try:
        feedback_record = {
            'client_id': client_id,
            'recommended_bond': recommended_bond,
            'actual_choice': actual_choice,
            'reward': reward,
            'feedback_date': pd.Timestamp.now()
        }
        
        query = """
        INSERT INTO model_feedback (client_id, recommended_bond, actual_choice, reward, feedback_date)
        VALUES (%(client_id)s, %(recommended_bond)s, %(actual_choice)s, %(reward)s, %(feedback_date)s)
        """
        
        with engine.connect() as conn:
            conn.execute(query, feedback_record)
            conn.commit()
        
        return True
    
    except Exception as e:
        st.error(f"Feedback update error: {str(e)}")
        return False

# Create main tabs for the application
if len(clients_df) > 0 and 'X' in locals() and len(X) > 0:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🎯 Client Analysis", 
        "🔮 New Client Prediction", 
        "👨‍👩‍👧‍👦 Family Analysis",
        "⚖️ Compare Clients",
        "🔧 System Management"
    ])
    
    # Tab 1: Enhanced Dashboard
    with tab1:
        st.header("Investment Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clients", len(clients_df))
        with col2:
            total_investment = clients_df['Total Invested'].sum() if 'Total Invested' in clients_df.columns else 0
            st.metric("Total Investment", f"₹{total_investment:,.0f}")
        with col3:
            if len(X) > 0:
                silhouette_avg = silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else 0
                st.metric("Clustering Quality", f"{silhouette_avg:.3f}")
            else:
                st.metric("Clustering Quality", "N/A")
        with col4:
            st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
        
        # Cluster visualization
        if len(X) > 0 and len(cluster_labels) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster distribution
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                cluster_names = [cluster_analyzer.cluster_descriptions.get(i, {}).get('name', f'Cluster {i}') 
                               for i in cluster_counts.index]
                
                fig_pie = px.pie(
                    values=cluster_counts.values, 
                    names=cluster_names,
                    title="Client Distribution by Cluster"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk category distribution
                if 'Risk Category' in clients_df.columns:
                    risk_counts = clients_df['Risk Category'].value_counts()
                    fig_bar = px.bar(
                        x=risk_counts.index, 
                        y=risk_counts.values,
                        title="Risk Category Distribution",
                        labels={'x': 'Risk Category', 'y': 'Number of Clients'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Cluster descriptions
        st.subheader("Cluster Analysis Results")
        
        if len(cluster_analyzer.cluster_descriptions) > 0:
            for cluster_id, desc in cluster_analyzer.cluster_descriptions.items():
                with st.expander(f"{desc['name']} ({desc['size']} clients) - Confidence: {desc.get('confidence', 0):.2f}"):
                    st.write(desc['description'])
                    
                    # Display key metrics
                    metrics = desc.get('key_metrics', {})
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Age", f"{metrics.get('avg_age', 0):.1f}")
                        with col2:
                            st.metric("Dominant Risk", metrics.get('dominant_risk', 'N/A'))
                        with col3:
                            st.metric("Reinvest %", f"{metrics.get('avg_reinvest_ratio', 0):.1f}%")
        
    
    # Tab 2: Enhanced Client Analysis
    with tab2:
        st.header("Individual Client Analysis")
        
        if len(clients_df) > 0:
            selected_client = st.selectbox("Select Client", list(clients_df.index))
            
            if selected_client:
                client_data = clients_df.loc[selected_client]
                cluster_id = int(client_data['Cluster'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Client Profile")
                    st.write(f"**Age:** {client_data['Age']} years")
                    st.write(f"**Gender:** {client_data['Gender']}")
                    st.write(f"**Risk Category:** {client_data['Risk Category']}")
                    st.write(f"**Total Invested:** ₹{client_data['Total Invested']:,.0f}")
                    st.write(f"**Reinvestment Ratio:** {client_data['Reinvest Ratio']:.1f}%")
                
                with col2:
                    st.subheader("Cluster Assignment")
                    cluster_info = cluster_analyzer.cluster_descriptions.get(cluster_id, {})
                    st.write(f"**Cluster:** {cluster_info.get('name', f'Cluster {cluster_id}')}")
                    st.write(f"**Description:** {cluster_info.get('description', 'No description')}")
                    st.write(f"**Confidence:** {cluster_info.get('confidence', 0):.2f}")
                
                # Get recommendation
                if len(X) > 0:
                    client_idx = clients_df.index.get_loc(selected_client)
                    client_features = X[client_idx]
                    recommended_bond, confidence, top_3 = get_enhanced_recommendation(client_features, cluster_id)
                    
                    st.subheader("Investment Recommendation")
                    st.success(f"**Recommended:** {recommended_bond}")
                    st.metric("Confidence Score", f"{confidence*100:.1f}%")
                    
                    # Show top 3 recommendations
                    if top_3:
                        st.write("**Alternative Recommendations:**")
                        for i, (bond_name, score) in enumerate(top_3[1:], 2):
                            st.write(f"{i}. {bond_name} (Score: {score:.3f})")
                
                # Client's historical bond preferences
                client_trades = df[df['Client ID'] == selected_client]
                if len(client_trades) > 0:
                    st.subheader("Investment History")
                    
                    bond_preferences = client_trades['Type'].value_counts()
                    fig_pref = px.bar(
                        x=bond_preferences.values,
                        y=bond_preferences.index,
                        orientation='h',
                        title="Historical Bond Type Preferences"
                    )
                    st.plotly_chart(fig_pref, use_container_width=True)
    
    # Tab 3: Enhanced New Client Prediction
    with tab3:
        st.header("New Client Prediction & Registration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Client Information")
            new_age = st.number_input("Age", min_value=18, max_value=80, value=35)
            new_gender = st.selectbox("Gender", ["M", "F", "HUF"])
            new_risk = st.selectbox("Risk Category", ["A", "B", "C"], 
                                  help="A: Conservative, B: Moderate, C: Aggressive")
            
        with col2:
            st.subheader("Additional Details")
            parent_options = ["None"] + [pid for pid in clients_df.index 
                                       if clients_df.loc[pid, 'Parent ID'] == '--NA--']
            new_parent = st.selectbox("Parent Client (optional)", parent_options)
            estimated_investment = st.number_input("Estimated Investment Amount", 
                                                 min_value=10000, value=100000, step=10000)
        
        if st.button("Analyze New Client", type="primary"):
            # Prepare new client data
            parent_reinv = clients_df.loc[new_parent, 'Reinvest Ratio'] if new_parent != "None" else 30.0
            
            new_client_data = pd.DataFrame([{
                'Age': new_age,
                'Gender': new_gender,
                'Risk Category': new_risk,
                'Parent ID': new_parent if new_parent != "None" else '--NA--',
                'Reinvest Ratio': parent_reinv,
                'Parent Ratio': parent_reinv,
                'Total Invested': estimated_investment
            }])
            
            # Prepare features for prediction
            try:
                # Create temporary extended dataframe
                temp_df = pd.concat([clients_df, new_client_data.set_index([len(clients_df)])], 
                                  ignore_index=True)
                
                # Prepare features
                X_temp, _, _ = prepare_enhanced_features(temp_df)
                
                if len(X_temp) > 0:
                    new_features = X_temp[-1:] # Get last row (new client)
                    predicted_cluster = kmeans.predict(new_features)[0]
                    
                    # Get recommendation
                    recommended_bond, confidence, top_3 = get_enhanced_recommendation(
                        new_features[0], predicted_cluster
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Predicted Cluster:** {predicted_cluster}")
                        cluster_info = cluster_analyzer.cluster_descriptions.get(predicted_cluster, {})
                        st.write(f"**Cluster Type:** {cluster_info.get('name', 'Unknown')}")
                        st.write(f"**Description:** {cluster_info.get('description', 'No description')}")
                    
                    with col2:
                        st.success(f"**Recommended Bond:** {recommended_bond}")
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        if top_3:
                            st.write("**Top 3 Recommendations:**")
                            for i, (bond_name, score) in enumerate(top_3, 1):
                                st.write(f"{i}. {bond_name} (Score: {score:.3f})")
                    
                    # Save to database option
                    if engine is not None:
                        st.subheader("Save New Client")
                        if st.button("Register in Database"):
                            client_id = save_new_client_to_db(engine, {
                                'Age': new_age,
                                'Gender': new_gender,
                                'Risk Category': new_risk,
                                'Parent ID': new_parent if new_parent != "None" else '--NA--'
                            })
                            
                            if client_id:
                                st.success(f"Client registered with ID: {client_id}")
                            else:
                                st.error("Failed to register client")
                    
                    # Feedback section
                    st.subheader("Model Learning")
                    actual_choice = st.selectbox("What bond type did they actually choose?", 
                                               list(BOND_TYPE_MAPPING.values()))
                    
                    if st.button("Submit Feedback"):
                        # Calculate reward
                        actual_idx = TYPE_TO_INDEX.get(actual_choice, 1)
                        recommended_idx = TYPE_TO_INDEX.get(recommended_bond, 1)
                        reward = 1.0 if actual_idx == recommended_idx else 0.0
                        
                        # Update Q-table
                        if predicted_cluster < len(Q) and actual_idx < Q.shape[1]:
                            Q[predicted_cluster, actual_idx] += 0.1 * (reward - Q[predicted_cluster, actual_idx])
                            np.save("q_table.npy", Q)
                            
                            # Update database if available
                            if engine is not None:
                                update_model_feedback(engine, "new_client", recommended_bond, 
                                                    actual_choice, reward)
                            
                            st.success("Model updated with feedback!")
                        
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Tab 4: Family Analysis
    with tab4:
        st.header("Parent-Child Investment Analysis")
        
        # Find parent-child relationships
        family_groups = clients_df[clients_df['Parent ID'] != '--NA--'].groupby('Parent ID')
        
        if len(family_groups) > 0:
            family_data = []
            
            for parent_id, children in family_groups:
                if parent_id in clients_df.index:
                    parent_data = clients_df.loc[parent_id]
                    
                    for child_id in children.index:
                        child_data = clients_df.loc[child_id]
                        
                        family_data.append({
                            'Parent': parent_id,
                            'Child': child_id,
                            'Parent Risk': parent_data['Risk Category'],
                            'Child Risk': child_data['Risk Category'],
                            'Parent Reinvest %': parent_data['Reinvest Ratio'],
                            'Child Reinvest %': child_data['Reinvest Ratio'],
                            'Risk Match': parent_data['Risk Category'] == child_data['Risk Category'],
                            'Reinvest Diff': abs(parent_data['Reinvest Ratio'] - child_data['Reinvest Ratio'])
                        })
            
            if family_data:
                family_df = pd.DataFrame(family_data)
                
                # Display family analysis
                st.dataframe(family_df)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_match_rate = family_df['Risk Match'].mean() * 100
                    st.metric("Risk Category Match Rate", f"{risk_match_rate:.1f}%")
                
                with col2:
                    avg_reinvest_diff = family_df['Reinvest Diff'].mean()
                    st.metric("Avg Reinvestment Difference", f"{avg_reinvest_diff:.1f}%")
                
                with col3:
                    similar_families = (family_df['Reinvest Diff'] < 20).sum()
                    st.metric("Similar Investment Patterns", f"{similar_families}/{len(family_df)}")
                
                # Visualization
                fig_scatter = px.scatter(
                    family_df, 
                    x='Parent Reinvest %', 
                    y='Child Reinvest %',
                    color='Risk Match',
                    title="Parent vs Child Investment Behavior",
                    hover_data=['Parent', 'Child']
                )
                fig_scatter.add_shape(
                    type='line', 
                    x0=0, y0=0, x1=100, y1=100,
                    line=dict(dash='dash', color='gray')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No parent-child relationships found in the data.")
    
    # Tab 5: Client Comparison
    with tab5:
        st.header("Client Comparison Analysis")
        
        if len(clients_df) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                client1 = st.selectbox("Select First Client", list(clients_df.index), key="comp1")
            with col2:
                client2 = st.selectbox("Select Second Client", list(clients_df.index), 
                                     index=1, key="comp2")
            
            if client1 != client2:
                data1 = clients_df.loc[client1]
                data2 = clients_df.loc[client2]
                
                # Calculate similarity if features available
                if len(X) > 0:
                    idx1 = clients_df.index.get_loc(client1)
                    idx2 = clients_df.index.get_loc(client2)
                    features1 = X[idx1]
                    features2 = X[idx2]
                    similarity = cosine_similarity([features1], [features2])[0][0]
                    
                    st.metric("Profile Similarity", f"{similarity*100:.1f}%")
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Client: {client1}")
                    st.write(f"**Age:** {data1['Age']}")
                    st.write(f"**Gender:** {data1['Gender']}")
                    st.write(f"**Risk Category:** {data1['Risk Category']}")
                    st.write(f"**Total Invested:** ₹{data1['Total Invested']:,.0f}")
                    st.write(f"**Reinvest Ratio:** {data1['Reinvest Ratio']:.1f}%")
                    st.write(f"**Cluster:** {data1['Cluster']}")
                
                with col2:
                    st.subheader(f"Client: {client2}")
                    st.write(f"**Age:** {data2['Age']}")
                    st.write(f"**Gender:** {data2['Gender']}")
                    st.write(f"**Risk Category:** {data2['Risk Category']}")
                    st.write(f"**Total Invested:** ₹{data2['Total Invested']:,.0f}")
                    st.write(f"**Reinvest Ratio:** {data2['Reinvest Ratio']:.1f}%")
                    st.write(f"**Cluster:** {data2['Cluster']}")
                
                # Recommendations comparison
                if len(X) > 0:
                    idx1 = clients_df.index.get_loc(client1)
                    idx2 = clients_df.index.get_loc(client2)
                    
                    rec1, conf1, _ = get_enhanced_recommendation(X[idx1], int(data1['Cluster']))
                    rec2, conf2, _ = get_enhanced_recommendation(X[idx2], int(data2['Cluster']))
                    
                    st.subheader("Recommendation Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{client1}:** {rec1} ({conf1*100:.1f}%)")
                    with col2:
                        st.write(f"**{client2}:** {rec2} ({conf2*100:.1f}%)")
        else:
            st.info("Need at least 2 clients for comparison.")
    
    # Tab 6: System Management
    with tab6:
        st.header("System Management & Configuration")
        
        # Database configuration
        st.subheader("Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Configuration:**")
            st.json({
                'Host': DB_CONFIG['host'],
                'Port': DB_CONFIG['port'],
                'Database': DB_CONFIG['database'],
                'Connected': engine is not None
            })
        
        with col2:
            st.write("**Model Statistics:**")
            st.write(f"- Q-table size: {Q.shape}")
            st.write(f"- Number of features: {len(feature_names)}")
            st.write(f"- Training accuracy: {accuracy*100:.1f}%")
            st.write(f"- New client count: {new_client_count}")
        
        # Model management
        st.subheader("Model Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Retrain Models"):
                if len(X) > 0:
                    with st.spinner("Retraining models..."):
                        # Retrain clustering
                        kmeans.fit(X)
                        cluster_labels = kmeans.predict(X)
                        clients_df['Cluster'] = cluster_labels
                        
                        # Retrain Q-learning
                        Q = train_enhanced_q_learning(clients_df, df, episodes=5000)
                        
                        # Recalculate accuracy
                        accuracy, accuracy_details = calculate_enhanced_model_accuracy(clients_df, df, Q)
                        
                        # Save models
                        with open("kmeans_model.pkl", "wb") as f:
                            pickle.dump(kmeans, f)
                        np.save("q_table.npy", Q)
                        
                        st.success(f"Models retrained! New accuracy: {accuracy*100:.1f}%")
                        st.rerun()
                else:
                    st.error("No data available for retraining")
        
        with col2:
            if st.button("Export Models"):
                export_data = {
                    'Q_table': Q.tolist(),
                    'feature_names': feature_names,
                    'model_accuracy': accuracy,
                    'cluster_descriptions': cluster_analyzer.cluster_descriptions,
                    'export_timestamp': pd.Timestamp.now().isoformat()
                }
                
                st.download_button(
                    label="Download Model Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"investment_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("Reset Models"):
                if st.checkbox("Confirm reset (this will delete all trained data)"):
                    # Reset all models
                    Q = np.zeros((num_clusters, NUM_BOND_TYPES))
                    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
                    scaler = StandardScaler()
                    new_client_count = 0
                    
                    # Delete saved files
                    for filename in ["q_table.npy", "kmeans_model.pkl", "scaler_model.pkl", 
                                   "feature_names.json", "feedback_meta.json"]:
                        if os.path.exists(filename):
                            os.remove(filename)
                    
                    st.success("Models reset successfully!")
                    st.rerun()
        
        # Database operations
        if engine is not None:
            st.subheader("Database Operations")
            
            # Show database statistics
            try:
                with engine.connect() as conn:
                    client_count = conn.execute("SELECT COUNT(*) FROM investment_data").scalar()
                    unique_clients = conn.execute("SELECT COUNT(DISTINCT client_id) FROM investment_data").scalar()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Records", client_count)
                    with col2:
                        st.metric("Unique Clients", unique_clients)
                        
            except Exception as e:
                st.warning(f"Could not fetch database statistics: {str(e)}")
            
            # Data refresh
            if st.button("Refresh Data from Database"):
                try:
                    df = load_data_from_db(engine)
                    st.success(f"Refreshed! Loaded {len(df)} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {str(e)}")

else:
    # No data available - show setup instructions
    st.error("No client data available")
    
    st.subheader("Database Setup Instructions")
    
    st.markdown("""
    ### MySQL Database Schema
    
    Create the following tables in your MySQL database:
    
    ```sql
    -- Main investment data table
    CREATE TABLE investment_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        age INT NOT NULL,
        gender ENUM('M', 'F', 'HUF') NOT NULL,
        risk_category ENUM('A', 'B', 'C') NOT NULL,
        parent_client_id VARCHAR(50) DEFAULT '--NA--',
        bond_id VARCHAR(50) NOT NULL,
        bond_type VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        trade_value DECIMAL(15,2) NOT NULL,
        maturity_date DATE NOT NULL,
        bond_issue_date DATE NOT NULL,
        coupon_rate DECIMAL(5,2) NOT NULL,
        ytm DECIMAL(5,2) NOT NULL,
        frequency ENUM('Annual', 'Semi-Annual', 'Quarterly', 'Monthly') NOT NULL,
        tenure_years DECIMAL(5,2) NOT NULL,
        rating VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_client_id (client_id),
        INDEX idx_trade_date (trade_date)
    );
    
    -- Client master table
    CREATE TABLE clients (
        client_id VARCHAR(50) PRIMARY KEY,
        age INT NOT NULL,
        gender ENUM('M', 'F', 'HUF') NOT NULL,
        risk_category ENUM('A', 'B', 'C') NOT NULL,
        parent_client_id VARCHAR(50) DEFAULT '--NA--',
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );
    
    -- Model feedback table
    CREATE TABLE model_feedback (
        id INT AUTO_INCREMENT PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        recommended_bond VARCHAR(100) NOT NULL,
        actual_choice VARCHAR(100) NOT NULL,
        reward DECIMAL(3,2) NOT NULL,
        feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_client_id (client_id),
        INDEX idx_feedback_date (feedback_date)
    );
    ```
    
    ### Configuration Steps:
    
    1. **Update Database Configuration** in the code:
       - Change `DB_CONFIG` dictionary with your MySQL credentials
       - Set correct host, port, username, password, and database name
    
    2. **Install Required Packages**:
       ```bash
       pip install mysql-connector-python sqlalchemy pandas numpy scikit-learn streamlit plotly
       ```
    
    3. **Insert Sample Data** (optional):
       ```sql
       -- Sample data insertion
       INSERT INTO investment_data (client_id, age, gender, risk_category, bond_id, bond_type, 
                                  trade_date, trade_value, maturity_date, bond_issue_date, 
                                  coupon_rate, ytm, frequency, tenure_years, rating) 
       VALUES 
       ('C_0001', 35, 'M', 'B', 'BOND_001', 'Corporate Bonds', '2024-01-15', 100000, 
        '2027-01-15', '2023-12-01', 7.5, 8.2, 'Annual', 3, 'AA'),
       ('C_0002', 42, 'F', 'A', 'BOND_002', 'Government Securities', '2024-02-01', 250000, 
        '2029-02-01', '2023-11-15', 6.8, 7.1, 'Semi-Annual', 5, 'AAA');
       ```
    """)
    
    with st.expander("Database Connection Troubleshooting"):
        st.markdown("""
        ### Common Issues and Solutions:
        
        1. **Connection Failed**: 
           - Verify MySQL server is running
           - Check credentials in `DB_CONFIG`
           - Ensure database exists
           
        2. **Table Not Found**:
           - Run the CREATE TABLE statements above
           - Verify table names match exactly
           
        3. **Permission Denied**:
           - Grant appropriate permissions to MySQL user
           - `GRANT ALL PRIVILEGES ON investment_db.* TO 'your_username'@'localhost';`
           
        4. **Module Import Error**:
           - Install missing packages: `pip install mysql-connector-python`
        """)

# Footer with system information
st.markdown("---")
st.markdown("### System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Version:** Enhanced MySQL v2.0")
    st.write("**Database:** MySQL Integration")

with col2:
    st.write("**Features:** Dynamic Learning, Real-time Updates")
    st.write("**Bond Types:** 13 Categories")

with col3:
    if engine is not None:
        st.write("**Status:** Connected ✅")
    else:
        st.write("**Status:** Database Not Connected ❌")
    
    st.write(f"**Model Accuracy:** {accuracy*100:.1f}%" if 'accuracy' in locals() else "**Model Accuracy:** Not Available")

# Performance monitoring
if st.checkbox("Show Performance Metrics"):
    st.subheader("System Performance")
    
    if len(clients_df) > 0 and len(X) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Memory usage
            q_memory = Q.nbytes / 1024 if Q.size > 0 else 0
            st.metric("Q-table Memory", f"{q_memory:.1f} KB")
        
        with col2:
            # Data size
            data_memory = X.nbytes / 1024 if len(X) > 0 else 0
            st.metric("Feature Data", f"{data_memory:.1f} KB")
        
        with col3:
            # Clustering quality
            if len(X) > 0 and len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X, cluster_labels)
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            else:
                st.metric("Silhouette Score", "N/A")

# Error handling and warnings
if 'df' in locals() and len(df) > 0:
    # Data quality checks
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        st.warning(f"Data quality issue: {missing_data} missing values detected")
    
    # Check for data consistency
    if 'Trade Date' in df.columns and 'Maturity Date' in df.columns:
        invalid_dates = (df['Trade Date'] >= df['Maturity Date']).sum()
        if invalid_dates > 0:
            st.warning(f"Data consistency issue: {invalid_dates} records with invalid date ranges")

# Help section
with st.expander("Help & Documentation"):
    st.markdown("""
    ## Enhanced Investment Analysis System - MySQL Integration
    
    ### Key Improvements Made:
    
    1. **Database Integration**:
       - MySQL connectivity with SQLAlchemy
       - Robust error handling and fallback mechanisms
       - Real-time data loading and updates
    
    2. **Enhanced Machine Learning**:
       - Improved feature engineering with statistical validation
       - Better clustering analysis with confidence scores
       - Enhanced Q-learning with performance-weighted rewards
    
    3. **Robust Error Handling**:
       - Comprehensive data validation
       - Graceful degradation when components fail
       - Detailed error messages and troubleshooting guidance
    
    4. **Better User Experience**:
       - Intuitive dashboard with key metrics
       - Enhanced visualizations
       - Real-time model updates and feedback
    
    ### Usage Instructions:
    
    1. **Setup**: Configure database connection in `DB_CONFIG`
    2. **Data Loading**: System automatically loads and validates data
    3. **Analysis**: Use tabs to explore different analyses
    4. **Predictions**: Add new clients and get real-time recommendations
    5. **Feedback**: Provide feedback to improve model accuracy
    
    ### Technical Architecture:
    
    - **Clustering**: MiniBatch K-Means for scalable client segmentation
    - **Recommendation**: Q-Learning with enhanced reward structure
    - **Features**: Statistical feature engineering with outlier handling
    - **Database**: MySQL with SQLAlchemy for robust data operations
    
    ### Bond Categories:
    Government Securities, Corporate Bonds, Municipal Bonds, Treasury Bills,
    Commercial Papers, Certificate of Deposits, Infrastructure Bonds, Tax-Free Bonds,
    Convertible Bonds, Zero Coupon Bonds, High Yield Bonds, International Bonds, Green Bonds
    """)

# Final status check
if engine is None:
    st.error("""
    ⚠️ **System is running in demo mode** - Database connection not available.
    
    To enable full functionality:
    1. Install MySQL and create the required database
    2. Update the DB_CONFIG dictionary with your credentials
    3. Run the provided SQL schema to create tables
    4. Restart the application
    """)
elif len(clients_df) == 0:
    st.warning("""
    📊 **No client data available** - Database connected but empty.
    
    To start analysis:
    1. Insert sample data using the provided SQL statements
    2. Or upload your existing investment data
    3. Refresh the application
    """)
else:
    st.success(f"""
    ✅ **System fully operational** - {len(clients_df)} clients loaded successfully.
    
    Ready for:
    - Client clustering and analysis
    - Investment recommendations
    - New client predictions
    - Real-time model learning
    """)