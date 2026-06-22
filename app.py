import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv 
import os
load_dotenv()
from google import genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Agentic AI Credit Risk System",
    page_icon="🏦",
    layout="wide"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#2E0854,#4B0082,#6A0DAD);
}

/* Main Heading */
.main-title{
    text-align:center;
    color:#FFD700;
    font-size:42px;
    font-weight:bold;
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
    font-weight:bold;
}

/* Section Headings */
.section-title{
    color:#FFD700;
    font-size:30px;
    font-weight:bold;
    margin-bottom:20px;
}

/* Labels */
label{
    color:white !important;
    font-size:18px !important;
    font-weight:bold !important;
}

/* Input Text */
.stNumberInput input{
    color:black !important;
    font-size:18px !important;
    font-weight:bold !important;
}

/* Metric Cards */
.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 0px 15px rgba(0,0,0,0.3);
}

.metric-value{
    font-size:28px;
    font-weight:bold;
    color:#4B0082;
}

.metric-label{
    color:black;
    font-size:16px;
}

/* Button */
.stButton > button{
    background:#FFD700;
    color:black;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
    height:55px;
    width:100%;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================

model = joblib.load("loan_model.pkl")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
def generate_pdf(
    income,
    credit_score,
    loan_amount,
    years_employed,
    points,
    decision,
    approval_score,
    risk,
    ai_text
):
    

    pdf_file = "loan_report.pdf"
    st.write("PDF Created:", pdf_file)

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("Loan Assessment Report", styles['Title'])
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(f"Income: ₹{income}", styles['BodyText'])
    )

    content.append(
        Paragraph(f"Credit Score: {credit_score}", styles['BodyText'])
    )

    content.append(
        Paragraph(f"Loan Amount: ₹{loan_amount}", styles['BodyText'])
    )

    content.append(
        Paragraph(f"Years Employed: {years_employed}", styles['BodyText'])
    )

    content.append(
        Paragraph(f"Risk Points: {points}", styles['BodyText'])
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(f"Decision: {decision}", styles['BodyText'])
    )

    content.append(
        Paragraph(
            f"Approval Probability: {approval_score}%",
            styles['BodyText']
        )
    )

    content.append(
        Paragraph(f"Risk Category: {risk}", styles['BodyText'])
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph("AI Financial Advisor", styles['Heading2'])
    )

    content.append(
        Paragraph(ai_text, styles['BodyText'])
    )

    doc.build(content)

    return pdf_file

# ================= HEADER =================

st.markdown("""
<h1 style='text-align:center;color:#FFD700;'>
Agentic AI Credit Risk & Loan Advisory System
</h1>

<h4 style='text-align:center;color:white;'>
Machine Learning Powered Loan Approval & Risk Assessment Platform
</h4>
""", unsafe_allow_html=True)

st.write("")
st.write("")


# ================= INPUT SECTION =================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "<p class='section-title'>Applicant Financial Information</p>",
        unsafe_allow_html=True
    )

    income = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=50000
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        value=700
    )

    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=0,
        value=20000
    )

with col2:

    st.markdown(
        "<p class='section-title'>Employment & Risk Profile</p>",
        unsafe_allow_html=True
    )

    years_employed = st.number_input(
        "Years of Employment",
        min_value=0,
        value=5
    )

    points = st.number_input(
        "Customer Risk Points",
        min_value=0,
        value=60
    )

st.write("")
st.write("")

# ================= PREDICTION BUTTON =================

if st.button("Evaluate Loan Application"):

    data = pd.DataFrame({
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "years_employed": [years_employed],
        "points": [points]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # ================= DECISION =================

    decision = "APPROVED" if prediction else "REJECTED"

    # ================= RISK CATEGORY =================

    if probability >= 0.80:
        risk = "LOW RISK"
    elif probability >= 0.50:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    approval_score = round(probability * 100, 2)
    risk_score = round(100 - approval_score, 2)

    # ================= METRIC CARDS =================

    st.write("")
    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{decision}</div>
            <div class='metric-label'>Loan Decision</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{approval_score}%</div>
            <div class='metric-label'>Approval Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{risk}</div>
            <div class='metric-label'>Risk Category</div>
        </div>
        """, unsafe_allow_html=True)

    # ================= BAR CHART =================

    st.markdown(
        "<p class='section-title'>Risk Analysis Dashboard</p>",
        unsafe_allow_html=True
    )

    risk_df = pd.DataFrame({
        "Metric": [
            "Credit Score",
            "Income Strength",
            "Employment Stability"
        ],
        "Score": [
            credit_score / 10,
            min((income / max(loan_amount, 1)) * 20, 100),
            min(years_employed * 10, 100)
        ]
    })

    fig = px.bar(
        risk_df,
        x="Metric",
        y="Score",
        title="Applicant Risk Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= PIE CHART =================

    st.markdown(
        "<p class='section-title'>Approval vs Risk Analysis</p>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        [approval_score, risk_score],
        labels=["Approval", "Risk"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # ================= AI CREDIT ASSESSMENT =================

    st.markdown(
        "<p class='section-title'>AI Credit Assessment</p>",
        unsafe_allow_html=True
    )

    reasons = []

    if credit_score >= 700:
        reasons.append(
            "Strong credit score indicating responsible repayment behaviour."
        )
    else:
        reasons.append(
            "Credit score is below the preferred lending threshold."
        )

    if income > loan_amount:
        reasons.append(
            "Income comfortably supports the requested loan amount."
        )
    else:
        reasons.append(
            "Requested loan amount is relatively high compared to income."
        )

    if years_employed >= 3:
        reasons.append(
            "Stable employment history lowers lending risk."
        )
    else:
        reasons.append(
            "Limited employment history increases lending risk."
        )

    for reason in reasons:
        st.markdown(
            f"<p style='color:white;font-size:18px;'>✓ {reason}</p>",
            unsafe_allow_html=True
        )

    # ================= RECOMMENDATIONS =================

    st.markdown(
        "<p class='section-title'>Financial Recommendations</p>",
        unsafe_allow_html=True
    )

    recommendations = []

    if credit_score < 650:
        recommendations.append(
            "Improve credit score before applying for higher loan amounts."
        )

    if loan_amount > income:
        recommendations.append(
            "Consider requesting a smaller loan amount."
        )

    if years_employed < 2:
        recommendations.append(
            "Longer employment history may improve eligibility."
        )

    if points < 50:
        recommendations.append(
            "Risk points suggest additional review is recommended."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Applicant profile appears financially strong."
        )

    for rec in recommendations:
        st.markdown(
            f"<p style='color:#FFD700;font-size:18px;'>⚡ {rec}</p>",
            unsafe_allow_html=True
        )

    # ================= GEMINI AGENTIC AI =================
        

               # ================= GEMINI AGENTIC AI =================

    st.markdown(
        "<p class='section-title'>Agentic AI Financial Advisor</p>",
        unsafe_allow_html=True
    )

    prompt = f"""
    Analyze this loan application and provide financial advice.

    Income: {income}
    Credit Score: {credit_score}
    Loan Amount: {loan_amount}
    Years Employed: {years_employed}
    Risk Points: {points}

    Decision: {decision}
    Risk Category: {risk}
    Approval Score: {approval_score}%

    Give a short financial assessment and recommendation.
    """

    ai_text = "AI assessment unavailable."

    try:

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        ai_text = response.text

    except Exception as e:

        ai_text = f"""
        Applicant profile has been evaluated successfully.

        Loan Decision: {decision}
        Risk Category: {risk}
        Approval Score: {approval_score}%

        Maintain a healthy credit score and debt-to-income ratio for better future borrowing opportunities.
        """

        st.warning("Gemini unavailable. Using local AI advisor.")

    # ================= SHOW AI RESPONSE =================

    st.markdown(
        f"""
        <div style="
            color:white;
            font-size:18px;
            line-height:1.8;
            padding:15px;
            border:2px solid #FFD700;
            border-radius:10px;
        ">
        {ai_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================= PDF GENERATION =================

    pdf_file = generate_pdf(
        income,
        credit_score,
        loan_amount,
        years_employed,
        points,
        decision,
        approval_score,
        risk,
        ai_text
    )

    with open(pdf_file, "rb") as file:

        st.download_button(
            label="📄 Download Loan Assessment Report",
            data=file,
            file_name="Loan_Report.pdf",
            mime="application/pdf"
        )

    # ================= AI CONFIDENCE =================

    st.markdown(
        "<p class='section-title'>AI Confidence Analysis</p>",
        unsafe_allow_html=True
    )

    st.progress(int(approval_score))

    st.markdown(
        f"""
        <p style="
            color:white;
            font-size:18px;
            font-weight:bold;
        ">
        Agent Confidence Score: {approval_score}%
        </p>
        """,
        unsafe_allow_html=True
    )
 
# ================= FOOTER =================

st.markdown(
    "<p class='footer'>Built using Python, Streamlit, Scikit-Learn, Plotly, Gemini AI and Explainable AI Techniques</p>",
    unsafe_allow_html=True
)