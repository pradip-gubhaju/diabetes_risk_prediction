# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .risk-moderate { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .risk-high { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    .risk-very-high { background-color: #721c24; color: white; padding: 10px; border-radius: 5px; }
    .feature-importance { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .recommendation-box { background-color: #e7f3ff; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

class DiabetesRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            # In a real app, you would load your actual trained model
            # For demo purposes, we'll create a placeholder
            self.model = None
            self.scaler = StandardScaler()
            self.feature_names = [
                'age', 'bmi', 'hba1c', 'systolic_blood_pressure', 'diastolic_blood_pressure',
                'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 'waist_circumference_cm',
                'body_fat_percentage', 'height_cm', 'weight_kg', 'sex_1', 'past_history_hypertension',
                'past_history_hyperlipidemia', 'vigorous_exercise_days_per_week', 'moderate_exercise_days_per_week'
            ]
            self.loaded = True
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def predict_risk(self, patient_data):
        """Predict diabetes risk with detailed explanations"""
        if not self.loaded:
            return None
        
        # Simulate prediction (replace with actual model prediction)
        risk_factors = self._calculate_risk_factors(patient_data)
        base_risk = self._calculate_base_risk(patient_data)
        risk_percentage = min(95, max(5, base_risk + risk_factors))
        
        return self._create_detailed_prediction(risk_percentage, patient_data)
    
    def _calculate_risk_factors(self, patient_data):
        """Calculate risk based on patient factors"""
        risk_score = 0
        
        # Age factor
        age = patient_data.get('age', 40)
        if age >= 60: risk_score += 25
        elif age >= 45: risk_score += 15
        elif age >= 35: risk_score += 5
        
        # BMI factor
        bmi = patient_data.get('bmi', 25)
        if bmi >= 30: risk_score += 20
        elif bmi >= 25: risk_score += 10
        
        # HbA1c factor
        hba1c = patient_data.get('hba1c', 5.5)
        if hba1c >= 6.5: risk_score += 30
        elif hba1c >= 5.7: risk_score += 15
        
        # Blood pressure factor
        systolic_bp = patient_data.get('systolic_blood_pressure', 120)
        if systolic_bp >= 140: risk_score += 10
        
        # Cholesterol factors
        hdl = patient_data.get('hdl_cholesterol', 50)
        if hdl < 40: risk_score += 5
        
        triglycerides = patient_data.get('triglycerides', 150)
        if triglycerides >= 200: risk_score += 5
        
        # Lifestyle factors
        exercise_days = patient_data.get('vigorous_exercise_days_per_week', 0) + patient_data.get('moderate_exercise_days_per_week', 0)
        if exercise_days < 2: risk_score += 10
        elif exercise_days < 4: risk_score += 5
        
        return risk_score
    
    def _calculate_base_risk(self, patient_data):
        """Calculate base risk percentage"""
        base_risk = 10  # Base population risk
        
        # Adjust based on key factors
        key_factors = ['hba1c', 'bmi', 'age', 'systolic_blood_pressure']
        for factor in key_factors:
            value = patient_data.get(factor, 0)
            if factor == 'hba1c' and value > 5.7:
                base_risk += (value - 5.7) * 10
            elif factor == 'bmi' and value > 25:
                base_risk += (value - 25) * 1
            elif factor == 'age' and value > 40:
                base_risk += (value - 40) * 0.5
        
        return base_risk
    
    def _create_detailed_prediction(self, risk_percentage, patient_data):
        """Create detailed prediction with explanations"""
        # Risk category
        if risk_percentage < 20:
            risk_category = "Low Risk"
            color = "green"
            icon = "✅"
        elif risk_percentage < 50:
            risk_category = "Moderate Risk"
            color = "orange"
            icon = "⚠️"
        elif risk_percentage < 80:
            risk_category = "High Risk"
            color = "red"
            icon = "🔴"
        else:
            risk_category = "Very High Risk"
            color = "darkred"
            icon = "🚨"
        
        # Key contributing factors
        contributing_factors = self._identify_contributing_factors(patient_data)
        
        # Recommendations
        recommendations = self._generate_recommendations(patient_data, risk_category)
        
        # Action plan
        action_plan = self._generate_action_plan(risk_category)
        
        return {
            'risk_percentage': risk_percentage,
            'risk_category': risk_category,
            'color': color,
            'icon': icon,
            'contributing_factors': contributing_factors,
            'recommendations': recommendations,
            'action_plan': action_plan,
            'feature_importance': self._generate_feature_importance(patient_data)
        }
    
    def _identify_contributing_factors(self, patient_data):
        """Identify key factors contributing to risk"""
        factors = []
        
        if patient_data.get('hba1c', 0) >= 5.7:
            factors.append(("High HbA1c", "Elevated blood sugar levels indicate prediabetes or diabetes risk"))
        
        if patient_data.get('bmi', 0) >= 25:
            factors.append(("High BMI", "Overweight or obesity increases diabetes risk"))
        
        if patient_data.get('age', 0) >= 45:
            factors.append(("Age", "Risk increases with age, especially after 45"))
        
        if patient_data.get('systolic_blood_pressure', 0) >= 130:
            factors.append(("High Blood Pressure", "Hypertension is linked to diabetes risk"))
        
        if patient_data.get('past_history_hypertension', 0) == 1:
            factors.append(("Hypertension History", "Previous high blood pressure increases risk"))
        
        exercise_days = patient_data.get('vigorous_exercise_days_per_week', 0) + patient_data.get('moderate_exercise_days_per_week', 0)
        if exercise_days < 3:
            factors.append(("Low Physical Activity", "Insufficient exercise increases diabetes risk"))
        
        return factors
    
    def _generate_recommendations(self, patient_data, risk_category):
        """Generate personalized recommendations"""
        recommendations = []
        
        # General recommendations
        recommendations.append("Maintain regular health check-ups")
        recommendations.append("Monitor blood glucose levels periodically")
        
        # BMI-based recommendations
        bmi = patient_data.get('bmi', 25)
        if bmi >= 25:
            recommendations.append(f"Aim for 5-10% weight reduction (Current BMI: {bmi:.1f})")
            recommendations.append("Focus on balanced diet with portion control")
        
        # Exercise recommendations
        exercise_days = patient_data.get('vigorous_exercise_days_per_week', 0) + patient_data.get('moderate_exercise_days_per_week', 0)
        if exercise_days < 5:
            recommendations.append(f"Increase physical activity to 150+ minutes per week (Current: ~{exercise_days * 30} minutes)")
        
        # HbA1c specific
        hba1c = patient_data.get('hba1c', 5.5)
        if hba1c >= 5.7:
            recommendations.append(f"Focus on reducing sugar and refined carbohydrate intake (HbA1c: {hba1c}%)")
        
        # Risk category specific
        if risk_category in ["High Risk", "Very High Risk"]:
            recommendations.append("Consult with healthcare provider for comprehensive assessment")
            recommendations.append("Consider regular glucose monitoring")
        
        return recommendations
    
    def _generate_action_plan(self, risk_category):
        """Generate action plan based on risk category"""
        if risk_category == "Low Risk":
            return [
                "Annual health screening",
                "Maintain healthy lifestyle",
                "Regular physical activity",
                "Balanced diet"
            ]
        elif risk_category == "Moderate Risk":
            return [
                "6-month health check-ups",
                "Lifestyle modification program",
                "Weight management",
                "Increased physical activity",
                "Dietary counseling"
            ]
        elif risk_category == "High Risk":
            return [
                "3-month medical follow-up",
                "Comprehensive diabetes screening",
                "Structured exercise program",
                "Medical nutrition therapy",
                "Consider medication if indicated"
            ]
        else:  # Very High Risk
            return [
                "Immediate medical consultation",
                "Frequent glucose monitoring",
                "Intensive lifestyle intervention",
                "Pharmacological intervention",
                "Regular specialist follow-up"
            ]
    
    def _generate_feature_importance(self, patient_data):
        """Generate simulated feature importance"""
        features = {
            'HbA1c': min(30, max(0, (patient_data.get('hba1c', 5.5) - 4.5) * 10)),
            'BMI': min(25, max(0, (patient_data.get('bmi', 25) - 18.5) * 1.5)),
            'Age': min(20, max(0, (patient_data.get('age', 40) - 30) * 0.5)),
            'Blood Pressure': min(15, max(0, (patient_data.get('systolic_blood_pressure', 120) - 110) * 0.3)),
            'Exercise': max(0, 20 - (patient_data.get('vigorous_exercise_days_per_week', 0) + patient_data.get('moderate_exercise_days_per_week', 0)) * 3),
            'Family History': 10 if patient_data.get('past_history_hypertension', 0) == 1 else 5,
            'Cholesterol': min(10, max(0, (200 - patient_data.get('hdl_cholesterol', 50)) * 0.2))
        }
        
        # Normalize to 100%
        total = sum(features.values())
        if total > 0:
            features = {k: (v/total)*100 for k, v in features.items()}
        
        return features

def create_risk_gauge(risk_percentage):
    """Create a visual risk gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score", 'font': {'size': 24}},
        delta = {'reference': 20, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'green'},
                {'range': [20, 50], 'color': 'yellow'},
                {'range': [50, 80], 'color': 'orange'},
                {'range': [80, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance chart"""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        title="Key Risk Factors Contributing to Your Score",
        labels={'x': 'Contribution (%)', 'y': 'Risk Factors'}
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_traces(marker_color='steelblue')
    
    return fig

def main():
    # Initialize predictor
    predictor = DiabetesRiskPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar for user input
    st.sidebar.header("📋 Patient Information")
    
    # Personal Information
    st.sidebar.subheader("Personal Details")
    age = st.sidebar.slider("Age", 18, 100, 45)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    height = st.sidebar.number_input("Height (cm)", 100, 250, 170)
    weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2) if height > 0 else 25
    st.sidebar.metric("BMI", f"{bmi:.1f}")
    
    # Medical History
    st.sidebar.subheader("Medical History")
    past_history_hypertension = st.sidebar.selectbox("History of Hypertension", ["No", "Yes"])
    past_history_hyperlipidemia = st.sidebar.selectbox("History of High Cholesterol", ["No", "Yes"])
    family_history_diabetes = st.sidebar.selectbox("Family History of Diabetes", ["No", "Yes"])
    
    # Lab Results
    st.sidebar.subheader("Laboratory Results")
    hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 15.0, 5.5, 0.1)
    systolic_bp = st.sidebar.slider("Systolic Blood Pressure", 80, 200, 120)
    diastolic_bp = st.sidebar.slider("Diastolic Blood Pressure", 50, 130, 80)
    hdl_cholesterol = st.sidebar.slider("HDL Cholesterol", 20, 100, 50)
    ldl_cholesterol = st.sidebar.slider("LDL Cholesterol", 50, 250, 100)
    triglycerides = st.sidebar.slider("Triglycerides", 50, 500, 150)
    
    # Lifestyle Factors
    st.sidebar.subheader("Lifestyle Factors")
    vigorous_exercise = st.sidebar.slider("Vigorous Exercise (days/week)", 0, 7, 2)
    moderate_exercise = st.sidebar.slider("Moderate Exercise (days/week)", 0, 7, 3)
    smoking = st.sidebar.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
    alcohol = st.sidebar.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
    
    # Physical Measurements
    st.sidebar.subheader("Physical Measurements")
    waist_circumference = st.sidebar.slider("Waist Circumference (cm)", 50, 150, 85)
    body_fat_percentage = st.sidebar.slider("Body Fat Percentage", 5.0, 50.0, 25.0, 0.1)
    
    # Prepare patient data
    patient_data = {
        'age': age,
        'bmi': bmi,
        'hba1c': hba1c,
        'systolic_blood_pressure': systolic_bp,
        'diastolic_blood_pressure': diastolic_bp,
        'hdl_cholesterol': hdl_cholesterol,
        'ldl_cholesterol': ldl_cholesterol,
        'triglycerides': triglycerides,
        'waist_circumference_cm': waist_circumference,
        'body_fat_percentage': body_fat_percentage,
        'height_cm': height,
        'weight_kg': weight,
        'sex_1': 1 if sex == "Male" else 0,
        'past_history_hypertension': 1 if past_history_hypertension == "Yes" else 0,
        'past_history_hyperlipidemia': 1 if past_history_hyperlipidemia == "Yes" else 0,
        'vigorous_exercise_days_per_week': vigorous_exercise,
        'moderate_exercise_days_per_week': moderate_exercise
    }
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Diabetes Risk", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            # Get prediction
            prediction = predictor.predict_risk(patient_data)
            
            if prediction:
                # Display results in main area
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Risk Gauge
                    st.plotly_chart(create_risk_gauge(prediction['risk_percentage']), use_container_width=True)
                    
                    # Risk Category
                    risk_class = f"risk-{prediction['risk_category'].lower().replace(' ', '-')}"
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <h3>{prediction['icon']} {prediction['risk_category']}</h3>
                        <h2>Risk Score: {prediction['risk_percentage']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Contributing Factors
                    st.subheader("🔍 Key Contributing Factors")
                    for factor, explanation in prediction['contributing_factors']:
                        with st.expander(f"📌 {factor}"):
                            st.write(explanation)
                
                # Feature Importance
                st.subheader("📊 Risk Factor Analysis")
                st.plotly_chart(create_feature_importance_chart(prediction['feature_importance']), use_container_width=True)
                
                # Recommendations and Action Plan
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("💡 Personalized Recommendations")
                    for i, recommendation in enumerate(prediction['recommendations'], 1):
                        st.markdown(f"""
                        <div class="recommendation-box">
                            {i}. {recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    st.subheader("📝 Action Plan")
                    for i, action in enumerate(prediction['action_plan'], 1):
                        st.markdown(f"""
                        <div class="feature-importance">
                            {i}. {action}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed Report
                st.markdown("---")
                st.subheader("📄 Comprehensive Risk Assessment Report")
                
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    st.write("**Patient Summary:**")
                    st.write(f"- Age: {age} years")
                    st.write(f"- Sex: {sex}")
                    st.write(f"- BMI: {bmi:.1f} ({'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'})")
                    st.write(f"- HbA1c: {hba1c}% ({'Normal' if hba1c < 5.7 else 'Prediabetes' if hba1c < 6.5 else 'Diabetes'})")
                    st.write(f"- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg")
                    
                with report_col2:
                    st.write("**Lifestyle Assessment:**")
                    st.write(f"- Exercise: {vigorous_exercise + moderate_exercise} days/week")
                    st.write(f"- Smoking: {smoking}")
                    st.write(f"- Alcohol: {alcohol}")
                    st.write(f"- Waist Circumference: {waist_circumference} cm")
                
                # Risk Interpretation
                st.subheader("🎯 Risk Interpretation")
                if prediction['risk_category'] == "Low Risk":
                    st.success("""
                    **Interpretation:** Your current risk of developing diabetes is relatively low. 
                    Continue maintaining healthy lifestyle habits including regular exercise and balanced nutrition.
                    """)
                elif prediction['risk_category'] == "Moderate Risk":
                    st.warning("""
                    **Interpretation:** You have moderate risk factors for diabetes. 
                    Focus on lifestyle modifications including weight management, increased physical activity, 
                    and dietary improvements to reduce your risk.
                    """)
                elif prediction['risk_category'] == "High Risk":
                    st.error("""
                    **Interpretation:** You have several significant risk factors for diabetes. 
                    Immediate lifestyle interventions and medical consultation are recommended. 
                    Regular monitoring of blood glucose levels is advised.
                    """)
                else:
                    st.error("""
                    **Interpretation:** You have very high risk factors for diabetes. 
                    Urgent medical consultation and comprehensive lifestyle intervention are strongly recommended. 
                    Consider regular glucose monitoring and potential pharmacological intervention.
                    """)
                
                # Download Report
                st.markdown("---")
                st.subheader("💾 Download Report")
                
                # Create downloadable report
                report_text = f"""
                DIABETES RISK ASSESSMENT REPORT
                ==============================
                
                Patient Information:
                - Age: {age}
                - Sex: {sex}
                - BMI: {bmi:.1f}
                - HbA1c: {hba1c}%
                - Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
                
                Risk Assessment:
                - Risk Score: {prediction['risk_percentage']:.1f}%
                - Risk Category: {prediction['risk_category']}
                
                Key Contributing Factors:
                {chr(10).join([f"- {factor}: {explanation}" for factor, explanation in prediction['contributing_factors']])}
                
                Recommendations:
                {chr(10).join([f"- {rec}" for rec in prediction['recommendations']])}
                
                Action Plan:
                {chr(10).join([f"- {action}" for action in prediction['action_plan']])}
                
                Generated by Diabetes Risk Prediction System
                """
                
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_text,
                    file_name=f"diabetes_risk_report_{age}_{sex}.txt",
                    mime="text/plain"
                )
    
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Diabetes Risk Prediction System
        
        This advanced tool helps assess your risk of developing type 2 diabetes based on your 
        health profile, lifestyle factors, and medical history.
        
        ### How to use:
        1. **Fill in your information** in the sidebar on the left
        2. **Provide accurate health data** for best results
        3. **Click the 'Predict Diabetes Risk' button** to get your assessment
        4. **Review your personalized report** with recommendations
        
        ### What you'll get:
        - 🎯 **Personalized risk score** (percentage)
        - 📊 **Detailed factor analysis**
        - 💡 **Customized recommendations**
        - 📝 **Action plan** for risk reduction
        - 📄 **Comprehensive report** for download
        
        ### Important Notes:
        - This tool is for educational and screening purposes only
        - Always consult healthcare professionals for medical advice
        - Regular health check-ups are recommended
        - Lifestyle changes can significantly reduce diabetes risk
        """)
        
        # Sample risk factors information
        with st.expander("🔍 Understanding Diabetes Risk Factors"):
            st.markdown("""
            **Major Risk Factors for Type 2 Diabetes:**
            
            **Non-modifiable Factors:**
            - Age (risk increases after 45)
            - Family history of diabetes
            - Ethnic background
            - Previous gestational diabetes
            
            **Modifiable Factors:**
            - Overweight or obesity (BMI ≥ 25)
            - Physical inactivity
            - High blood pressure
            - Abnormal cholesterol levels
            - Prediabetes (HbA1c 5.7-6.4%)
            - Metabolic syndrome
            
            **Early detection and lifestyle modifications can prevent or delay type 2 diabetes onset.**
            """)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Global Diabetes Prevalence", "537 million", "16% increase")
        with col2:
            st.metric("Undiagnosed Cases", "~50%", "Early detection crucial")
        with col3:
            st.metric("Prevention Success", "58% reduction", "With lifestyle changes")

if __name__ == "__main__":
    main()
