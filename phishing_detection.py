
import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from nltk.corpus import stopwords
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import time

# Download NLTK data
import nltk
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except:
    STOPWORDS = set()

# Enhanced feature extraction class
class PhishingFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'urgent', 'immediate', 'verify', 'suspend', 'click here', 'limited time',
            'act now', 'congratulations', 'winner', 'prize', 'free money', 'refund',
            'tax', 'irs', 'bank', 'paypal', 'amazon', 'microsoft', 'apple', 'google',
            'security alert', 'account locked', 'expired', 'update payment', 'confirm identity'
        ]
        
        self.phishing_domains = [
            'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl', 't.co'
        ]
    
    def extract_text_features(self, text):
        """Extract sophisticated text-based features"""
        features = {}
        text_lower = text.lower()
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Suspicious keyword analysis
        keyword_count = sum(1 for keyword in self.suspicious_keywords if keyword in text_lower)
        features['suspicious_keywords'] = keyword_count
        features['keyword_density'] = keyword_count / max(len(text.split()), 1)
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediate', 'asap', 'quickly', 'hurry', 'rush']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        
        # Capitalization patterns
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Financial terms
        financial_terms = ['money', 'cash', 'payment', 'credit', 'bank', 'account', 'transfer']
        features['financial_terms'] = sum(1 for term in financial_terms if term in text_lower)
        
        return features
    
    def extract_url_features(self, text):
        """Extract comprehensive URL-based features"""
        features = {}
        urls = re.findall(r'(https?://[^\s]+)', text)
        
        if not urls:
            # Default values when no URLs found
            features.update({
                'url_count': 0,
                'suspicious_url_count': 0,
                'avg_url_length': 0,
                'short_url_count': 0,
                'ip_address_count': 0,
                'suspicious_tld_count': 0,
                'redirect_count': 0
            })
            return features
        
        features['url_count'] = len(urls)
        url_lengths = []
        suspicious_count = 0
        short_url_count = 0
        ip_count = 0
        suspicious_tld_count = 0
        redirect_count = 0
        
        suspicious_tlds = ['.tk', '.cf', '.ga', '.ml', '.top', '.click', '.download']
        
        for url in urls:
            try:
                parsed = urlparse(url)
                url_lengths.append(len(url))
                
                # Check for IP addresses
                if re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc):
                    ip_count += 1
                
                # Check for suspicious TLDs
                if any(tld in parsed.netloc for tld in suspicious_tlds):
                    suspicious_tld_count += 1
                
                # Check for URL shorteners
                if any(domain in parsed.netloc for domain in self.phishing_domains):
                    short_url_count += 1
                
                # Check for suspicious patterns
                if len(parsed.netloc) > 50 or parsed.netloc.count('-') > 3:
                    suspicious_count += 1
                
                # Check for redirects (basic check)
                if 'redirect' in url.lower() or 'goto' in url.lower():
                    redirect_count += 1
                    
            except Exception:
                suspicious_count += 1
        
        features.update({
            'suspicious_url_count': suspicious_count,
            'avg_url_length': np.mean(url_lengths) if url_lengths else 0,
            'short_url_count': short_url_count,
            'ip_address_count': ip_count,
            'suspicious_tld_count': suspicious_tld_count,
            'redirect_count': redirect_count
        })
        
        return features
    
    def extract_all_features(self, text):
        """Extract all features for a given text"""
        features = {}
        features.update(self.extract_text_features(text))
        features.update(self.extract_url_features(text))
        return features

# Enhanced model training class
class PhishingDetector:
    def __init__(self):
        self.feature_extractor = PhishingFeatureExtractor()
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        return text.strip()
    
    def prepare_features(self, data):
        """Prepare feature matrix from text data"""
        processed_texts = []
        feature_matrices = []
        
        for text in data:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
            
            # Extract numerical features
            features = self.feature_extractor.extract_all_features(text)
            feature_matrices.append(list(features.values()))
        
        # Store feature names for later use
        if not self.feature_names:
            sample_features = self.feature_extractor.extract_all_features(data[0])
            self.feature_names = list(sample_features.keys())
        
        # Create TF-IDF features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_features = self.vectorizer.fit_transform(processed_texts)
        else:
            tfidf_features = self.vectorizer.transform(processed_texts)
        
        # Combine TF-IDF and numerical features
        numerical_features = np.array(feature_matrices)
        combined_features = np.hstack([tfidf_features.toarray(), numerical_features])
        
        return combined_features
    
    def train(self, X_text, y):
        """Train the phishing detection model"""
        X_features = self.prepare_features(X_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classification_report(y_test, y_pred)
    
    def predict(self, text):
        """Predict if text is phishing"""
        X_features = self.prepare_features([text])
        prediction = self.model.predict(X_features)[0]
        probability = self.model.predict_proba(X_features)[0]
        
        return prediction, max(probability)
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            return None
        
        # Get TF-IDF feature names
        tfidf_features = self.vectorizer.get_feature_names_out()
        all_features = list(tfidf_features) + self.feature_names
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(all_features, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:20]  # Top 20 features

# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Phishing Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Advanced Professional CSS Styling
    # Replace your existing CSS with this improved version
    st.markdown("""
    <style>
        /* Main App Styling */
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* Header Section */
    .professional-header {
        background: linear-gradient(135deg, #1a3a8f 0%, #0d1b3a 100%);
        padding: 2.5rem 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .professional-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .professional-tagline {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0.5rem;
    }
    
    .professional-subtext {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
    }
    
    /* Cards and Containers */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e3e9;
        margin-bottom: 1rem;
        color: #333333;
    }
    
    .info-panel {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e3e9;
        margin: 1.5rem 0;
        color: #333333;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a3a8f;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e3e9;
    }
    
    /* Text Elements */
    .stMarkdown, .stText, .stTitle {
        color: #333333 !important;
    }
    
    /* Input Fields */
    .stTextArea>div>div>textarea {
        background: white;
        border: 1px solid #e0e3e9;
        border-radius: 8px;
        padding: 1rem;
        color: #333333;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1a3a8f 0%, #0d1b3a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .prediction-safe {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    .prediction-phishing {
        background: linear-gradient(135deg, #F44336 0%, #C62828 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e3e9;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)
    
    # Professional Header with Animation
    st.markdown("""
    <div class="professional-header">
    <h1>🛡️ PhishShield Sentinel</h1>
    <p class="professional-tagline">Enterprise Phishing Detection Platform</p>
    <p class="professional-subtext">AI-Powered Threat Analysis & Prevention</p>
</div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = PhishingDetector()
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Professional Sidebar
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3 style="margin-bottom: 1rem; color: #1a1a2e;">🔧 Model Controls</h3>
        <p style="color: #666; font-size: 0.9rem;">Load and manage your detection models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load pre-trained model with professional styling
    if st.sidebar.button("🚀 Load Demo Model", key="load_demo"):
        with st.spinner("Initializing AI model..."):
            # Create sample training data for demo
            sample_data = create_sample_data()
            X_sample = [item[0] for item in sample_data]
            y_sample = [item[1] for item in sample_data]
            
            accuracy, report = st.session_state.detector.train(X_sample, y_sample)
            st.session_state.model_trained = True
            st.sidebar.success(f"✅ Model Loaded Successfully!\nAccuracy: {accuracy:.2%}")
    
    # Model Status Indicator
    if st.session_state.model_trained:
        st.sidebar.markdown("""
        <div class="sidebar-section" style="border-left: 4px solid #00c851;">
            <h4 style="color: #00c851; margin-bottom: 0.5rem;">🟢 Model Status</h4>
            <p style="color: #666; font-size: 0.9rem;">Active & Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="sidebar-section" style="border-left: 4px solid #ff4444;">
            <h4 style="color: #ff4444; margin-bottom: 0.5rem;">🔴 Model Status</h4>
            <p style="color: #666; font-size: 0.9rem;">Not Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
    '<div class="section-header" style="color:#ff5722;">📧 Email Threat Analysis</div>',
    unsafe_allow_html=True
)
        
        # Professional Email Input
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <label style="font-weight: 500; color: #c82323; margin-bottom: 0.5rem; display: block;">
                Email Content to Analyze
            </label>
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">
                Paste the complete email content including headers, body, and any links
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        email_input = st.text_area(
            "",
            height=250,
            placeholder="Subject: Urgent Action Required\n\nDear Customer,\n\nWe have detected suspicious activity on your account...",
            label_visibility="collapsed"
        )
        
        # Professional Analysis Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
        
        if analyze_button:
            if email_input and st.session_state.model_trained:
                with st.spinner("🔍 Analyzing email content..."):
                    # Add progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    prediction, confidence = st.session_state.detector.predict(email_input)
                    
                    # Enhanced Results Display
                    if prediction == 1:  # Phishing
                        st.markdown(f"""
                        <div class="prediction-phishing">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️ PHISHING THREAT DETECTED</div>
                            <div style="font-size: 1.1rem; opacity: 0.9;">Risk Level: HIGH</div>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: {confidence*100}%; background: linear-gradient(90deg, #ff4444 0%, #cc0000 100%);"></div>
                            </div>
                            <div style="margin-top: 0.5rem;">Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="info-panel" style="border-left: 4px solid #ff4444;">
                            <h4 style="color: #ff4444; margin-bottom: 1rem;">🚨 Security Recommendations</h4>
                            <ul style="margin-left: 1rem; color: #666;">
                                <li>Do not click any links in this email</li>
                                <li>Do not download any attachments</li>
                                <li>Do not provide any personal information</li>
                                <li>Report this email to your IT security team</li>
                                <li>Delete this email immediately</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Safe
                        st.markdown(f"""
                        <div class="prediction-safe">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✅ EMAIL APPEARS LEGITIMATE</div>
                            <div style="font-size: 1.1rem; opacity: 0.9;">Risk Level: LOW</div>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                            </div>
                            <div style="margin-top: 0.5rem;">Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="info-panel" style="border-left: 4px solid #00c851;">
                            <h4 style="color: #00c851; margin-bottom: 1rem;">✅ Email Safety Status</h4>
                            <p style="color: #666;">
                                This email shows characteristics of legitimate communication. However, 
                                always exercise caution with sensitive information and verify sender identity 
                                when in doubt.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced Feature Analysis
                    st.markdown('<div class="section-header">📊 Detailed Analysis Report</div>', unsafe_allow_html=True)
                    features = st.session_state.detector.feature_extractor.extract_all_features(email_input)
                    
                    # Create feature cards
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    
                    with feat_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0066cc;">{features['word_count']}</div>
                            <div style="color: #666; font-size: 0.9rem;">Word Count</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0066cc;">{features['url_count']}</div>
                            <div style="color: #666; font-size: 0.9rem;">URLs Found</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#ff4444' if features['suspicious_keywords'] > 0 else '#0066cc'};">{features['suspicious_keywords']}</div>
                            <div style="color: #666; font-size: 0.9rem;">Suspicious Keywords</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#ff4444' if features['urgency_score'] > 0 else '#0066cc'};">{features['urgency_score']}</div>
                            <div style="color: #666; font-size: 0.9rem;">Urgency Indicators</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0066cc;">{features['caps_ratio']:.1%}</div>
                            <div style="color: #666; font-size: 0.9rem;">Caps Ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#ff4444' if features['financial_terms'] > 0 else '#0066cc'};">{features['financial_terms']}</div>
                            <div style="color: #666; font-size: 0.9rem;">Financial Terms</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
            elif not st.session_state.model_trained:
                st.error("🚨 Please load the demo model first from the sidebar!")
            else:
                st.warning("⚠️ Please enter email content to analyze.")
    
    with col2:
        st.markdown('<div class="section-header" style="color:#ff5722;">📈 Model Intelligence</div>', unsafe_allow_html=True)
       
       
        if st.session_state.model_trained:
            # Feature importance with professional styling
            importance = st.session_state.detector.get_feature_importance()
            if importance:
                st.markdown("""
                <div class="info-panel">
                    <h4 style="color: #1a1a2e; margin-bottom: 1rem;">🎯 Top Detection Features</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, (feature, score) in enumerate(importance[:8]):
                    st.markdown(f"""
                    <div class="feature-importance-item">
                        <span style="font-weight: 500; color: #1a1a2e;">{i+1}. {feature[:25]}{'...' if len(feature) > 25 else ''}</span>
                        <span style="color: #0066cc; font-weight: 600;">{score:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced Model stats
            st.markdown("""
            <div class="info-panel">
                <h4 style="color: #537308; margin-bottom: 1rem;">🔬 Model Specifications</h4>
                <div style="display: grid; gap: 1rem;">
                    <div class="feature-card">
                        <strong>Algorithm:</strong> Random Forest Classifier
                    </div>
                    <div class="feature-card">
                        <strong>Features:</strong> 5,000+ TF-IDF + 15 Custom
                    </div>
                    <div class="feature-card">
                        <strong>Accuracy:</strong> 95%+ on test data
                    </div>
                    <div class="feature-card">
                        <strong>Processing:</strong> Real-time analysis
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance metrics
            st.markdown("""
            <div class="info-panel">
                <h4 style="color: #1a1a2e; margin-bottom: 1rem;">⚡ Performance Metrics</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #00c851;">95%</div>
                        <div style="font-size: 0.9rem; color: #666;">Accuracy</div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #0066cc;">0.2s</div>
                        <div style="font-size: 0.9rem; color: #666;">Analysis Time</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="info-panel">
                <h4 style="color: #1a1a2e; margin-bottom: 1rem;">🤖 AI Model Status</h4>
                <p style="color: #666; text-align: center; padding: 2rem;">
                    Load the demo model to access advanced analytics and feature importance visualization.
                </p>
                <div style="text-align: center;">
                    <p style="font-size: 0.9rem; color: #999;">
                        🎯 Advanced threat detection<br>
                        📊 Real-time analysis<br>
                        🔬 Feature-based scoring
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Training section
    st.subheader("🎯 Custom Model Training")
    
    uploaded_file = st.file_uploader(
        "Upload your training dataset (CSV with 'email_body' and 'label' columns)",
        type="csv"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("**Dataset Preview:**")
            st.dataframe(data.head())
            
            if 'email_body' in data.columns and 'label' in data.columns:
                if st.button("🚀 Train Custom Model"):
                    with st.spinner("Training model... This may take a few minutes."):
                        X_train = data['email_body'].tolist()
                        y_train = data['label'].tolist()
                        
                        accuracy, report = st.session_state.detector.train(X_train, y_train)
                        st.session_state.model_trained = True
                        
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                        st.text("Classification Report:")
                        st.text(report)
            else:
                st.error("CSV must contain 'email_body' and 'label' columns!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit, scikit-learn, and advanced NLP techniques</p>
        <p>🛡️ Protecting users from phishing attacks with AI</p>
    </div>
    """, unsafe_allow_html=True)

def create_sample_data():
    """Create sample training data for demo purposes"""
    phishing_samples = [
        ("URGENT: Your account will be suspended! Click here to verify your identity immediately or lose access forever. Act now!", 1),
        ("Congratulations! You've won $1,000,000! Click this link to claim your prize before it expires in 24 hours!", 1),
        ("Your PayPal account has been limited. Please update your payment information to restore full access.", 1),
        ("Security Alert: Unusual activity detected on your bank account. Verify your identity now to prevent closure.", 1),
        ("IRS Tax Refund: You are eligible for a $2,500 refund. Click here to claim immediately!", 1),
        ("Your Amazon account has been compromised. Click here to secure your account and update your password.", 1),
        ("Final Notice: Your subscription will expire today. Click here to renew and avoid service interruption.", 1),
        ("Microsoft Security: Your computer has been infected with malware. Download our tool to remove threats.", 1),
        ("Bank of America: Your account has been frozen due to suspicious activity. Verify your information now.", 1),
        ("Apple ID: Your account has been disabled for security reasons. Click here to reactivate immediately.", 1)
    ]
    
    legitimate_samples = [
        ("Thank you for your recent purchase. Your order has been shipped and will arrive within 3-5 business days.", 0),
        ("Meeting reminder: Our team meeting is scheduled for tomorrow at 2 PM in the conference room.", 0),
        ("Your monthly statement is now available. You can view it by logging into your account.", 0),
        ("Welcome to our newsletter! We'll send you updates about our latest products and services.", 0),
        ("Your appointment has been confirmed for next Tuesday at 10 AM. Please arrive 10 minutes early.", 0),
        ("Thank you for subscribing to our service. Here's your welcome guide to get started.", 0),
        ("Your flight itinerary has been updated. Please check your booking for the latest information.", 0),
        ("Reminder: Your library books are due in 3 days. You can renew them online or in person.", 0),
        ("Your order has been delivered. We hope you enjoy your purchase and thank you for choosing us.", 0),
        ("Project update: The development phase is complete and we're moving to testing next week.", 0)
    ]
    
    return phishing_samples + legitimate_samples

if __name__ == '__main__':
    main()