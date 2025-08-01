import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and encoders
sgpa_model = joblib.load("sgpa_model.pkl")
risk_model = joblib.load("risk_model.pkl")
weak_model = joblib.load("weak_model.pkl")
label_encoder_risk = joblib.load("label_encoder_risk.pkl")
label_encoder_weak = joblib.load("label_encoder_weak.pkl")

# Prediction function
def predict_student(data):
    sgpa = sgpa_model.predict(data)[0]
    risk = label_encoder_risk.inverse_transform(risk_model.predict(data))[0]
    weak = label_encoder_weak.inverse_transform(weak_model.predict(data))[0]
    return sgpa, risk, weak

# Title
st.title("🎓 Academic Performance Predictor")

# Tabs
tabs = st.tabs(["Single Student Prediction", "Bulk Prediction via CSV"])

# --- SINGLE ---
with tabs[0]:
    st.subheader("👤 Enter Student Details")

    col1, col2 = st.columns(2)
    with col1:
        previous_sgpa = st.slider("Previous SGPA", 0.0, 10.0, 7.5)
        avg_programming_score = st.slider("Avg Programming Score", 0, 100, 60)
        avg_practical_score = st.slider("Avg Practical Score", 0, 100, 70)
    with col2:
        avg_conceptual_score = st.slider("Avg Conceptual Score", 0, 100, 75)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        job_hours = st.slider("Job Hours per Week", 0, 40, 5)

    if st.button("📌 Predict"):
        input_df = pd.DataFrame([{
            "previous_sgpa": previous_sgpa,
            "avg_programming_score": avg_programming_score,
            "avg_practical_score": avg_practical_score,
            "avg_conceptual_score": avg_conceptual_score,
            "attendance": attendance,
            "job_hours": job_hours
        }])

        sgpa, risk, weak = predict_student(input_df)

        st.success("Prediction Complete")
        st.metric("📈 Predicted SGPA", f"{sgpa:.2f}")
        st.write(f"⚠️ **Risk Level:** {risk}")
        st.write(f"📚 **Weak Course Type:** {weak}")

        # Simulated SGPA chart (previous + predicted)
        chart_data = pd.DataFrame({
            "Semester": ["Prev", "Predicted"],
            "SGPA": [previous_sgpa, sgpa]
        })
        st.line_chart(chart_data.set_index("Semester"))

# --- BULK ---
with tabs[1]:
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'student_id' in df.columns:
            df = df.drop(columns=['student_id'])
        st.write("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        if st.button("🚀 Run Bulk Prediction"):
            sgpa_preds = sgpa_model.predict(df)
            risk_preds = label_encoder_risk.inverse_transform(risk_model.predict(df))
            weak_preds = label_encoder_weak.inverse_transform(weak_model.predict(df))

            result_df = df.copy()
            result_df["Predicted SGPA"] = sgpa_preds
            result_df["Risk Level"] = risk_preds
            result_df["Weak Course Type"] = weak_preds

            st.success("✅ Predictions Generated")
            st.dataframe(result_df)

            # --- Graphs ---
            st.subheader("📊 Visual Summary")

            # SGPA Line Chart
            st.write("**SGPA Trend Across Students**")
            sgpa_chart = pd.DataFrame({"Student Index": range(len(sgpa_preds)), "SGPA": sgpa_preds})
            st.line_chart(sgpa_chart.set_index("Student Index"))

            # Risk Pie Chart
            st.write("**Risk Level Distribution**")
            risk_counts = pd.Series(risk_preds).value_counts()
            st.plotly_chart({
                "data": [{"labels": risk_counts.index, "values": risk_counts.values, "type": "pie"}],
                "layout": {"margin": dict(t=0, b=0, l=0, r=0)}
            })

            # Weak Course Bar Chart
            st.write("**Weak Course Type Count**")
            weak_counts = pd.Series(weak_preds).value_counts()
            st.bar_chart(weak_counts)

            # Preventive Tip
            most_common_weak = weak_counts.idxmax()
            st.info(f"🔍 **Preventive Tip:** Most students are weak in **{most_common_weak}**. Consider offering extra support or workshops for this course type.")

            if "previous_sgpa" in df.columns and len(df) == len(sgpa_preds):
                sgpa_decline_ratio = (df["previous_sgpa"] > sgpa_preds).mean()

                if sgpa_decline_ratio > 0.5:  # Customize threshold if needed
                    st.warning(f"📉 **Alert:** {sgpa_decline_ratio:.0%} of students show a decline in SGPA compared to the previous semester.")
                    st.info("""
                    🛠 **Suggested Actions to Address SGPA Decline**
                    - Schedule academic counseling sessions for affected students.
                    - Review course load and check for overburdening.
                    - Encourage regular self-assessment through quizzes and mock exams.
                    - Promote peer mentoring or tutoring programs.
                    """)

            # CSV download
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Result CSV",
                data=csv,
                file_name="predicted_results.csv",
                mime="text/csv"
            )
