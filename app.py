import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import cohere

# Load ML model and label encoder
model = joblib.load("random_forest_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define skills and interests
streams = ['Science', 'Commerce', 'Arts']

skills_dict = {
    'Science': ['Python', 'Math', 'Machine Learning', 'Physics', 'Biology', 'Chemistry', 'Data Analysis'],
    'Commerce': ['Accounting', 'Excel', 'Tally', 'Economics', 'Finance', 'Business Analytics'],
    'Arts': ['Writing', 'Design', 'Photoshop', 'Public Speaking', 'Creativity', 'Photography']
}

interests_dict = {
    'Science': ['AI', 'Research', 'Technology', 'Healthcare', 'Space', 'Engineering'],
    'Commerce': ['Finance', 'Business', 'Marketing', 'Management', 'Startups'],
    'Arts': ['Storytelling', 'Media', 'Art', 'Fashion', 'Literature', 'Social Work']
}

# Initialize Cohere
co = cohere.Client(st.secrets["cohere"]["cohere_api_key"])

# Prediction function
def predict_career(stream, skills, interests):
    combined_text = stream + " " + " ".join(skills + interests)
    probs = model.predict_proba([combined_text])[0]
    top_3_idx = np.argsort(probs)[-3:][::-1]
    careers = label_encoder.inverse_transform(top_3_idx)
    confidences = probs[top_3_idx] * 100
    return list(zip(careers, confidences))

# Log predictions
def log_prediction(stream, skills, interests, prediction):
    log_data = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Stream": [stream],
        "Skills": [", ".join(skills)],
        "Interests": [", ".join(interests)],
        "Top Career": [prediction[0][0]],
        "Confidence": [f"{prediction[0][1]:.2f}%"]
    }
    df_log = pd.DataFrame(log_data)
    if os.path.exists("predictions_log.csv"):
        df_log.to_csv("predictions_log.csv", mode='a', header=False, index=False)
    else:
        df_log.to_csv("predictions_log.csv", index=False)

# Generate report with Cohere
def generate_report(career):
    prompt = f"""
Generate a detailed career report for someone interested in becoming a {career}. Include:

1. What is a {career}?
2. Top colleges in India offering courses related to this career
3. Average salary in INR
4. Pros of choosing this career
5. Cons or challenges
Make it sound personal and helpful for students.
"""
    response = co.generate(prompt=prompt, max_tokens=1000)
    return response.generations[0].text.strip()

# Send email
def send_email(receiver_email, career, report_text):
    message = Mail(
        from_email=st.secrets["sendgrid"]["from_email"],
        to_emails=receiver_email,
        subject=f"Your Career Report: {career}",
        plain_text_content=report_text
    )
    try:
        sg = SendGridAPIClient(st.secrets["sendgrid"]["api_key"])
        response = sg.send(message)
        return response.status_code == 202
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")
        return False

# Streamlit App UI
st.set_page_config(page_title="AI Career Predictor", layout="centered")
st.title("🎓 AI Career Predictor for Indian Students")

tab1, tab2 = st.tabs(["📊 Predict Your Career", "📈 Analytics Dashboard"])

with tab1:
    st.subheader("Tell us about yourself")
    stream = st.selectbox("Choose your stream", streams)
    selected_skills = st.multiselect("Select your skills", skills_dict[stream])
    selected_interests = st.multiselect("Select your interests", interests_dict[stream])

    if "predictions" not in st.session_state:
        st.session_state.predictions = []

    if st.button("Predict Career"):
        if not selected_skills or not selected_interests:
            st.warning("Please select at least one skill and one interest.")
        else:
            predictions = predict_career(stream, selected_skills, selected_interests)
            st.session_state.predictions = predictions
            top_career = predictions[0][0]
            st.success(f"🎯 Top Career: **{top_career}** ({predictions[0][1]:.2f}%)")

            st.write("📌 Other Suggestions:")
            for career, score in predictions[1:]:
                st.write(f"- {career} ({score:.2f}%)")

            log_prediction(stream, selected_skills, selected_interests, predictions)

    if st.session_state.predictions:
        st.markdown("---")
        st.subheader("📨 Get Your Career Report on Email")
        email = st.text_input("Enter your email")
        if st.button("📬 Send My Career Report"):
            top_career = st.session_state.predictions[0][0]
            with st.spinner("Generating report..."):
                try:
                    report = generate_report(top_career)
                    success = send_email(email, top_career, report)
                    if success:
                        st.success("✅ Career report sent successfully!")
                    else:
                        st.error("❌ Failed to send report.")
                except Exception as e:
                    st.error(f"❌ Error during report generation: {e}")

with tab2:
    st.subheader("📈 Prediction Logs")
    if os.path.exists("predictions_log.csv"):
        df = pd.read_csv("predictions_log.csv")
        st.dataframe(df.tail(10))

        st.write("### 🔝 Most Recommended Careers")
        top_careers = df["Top Career"].value_counts().head(5)
        fig, ax = plt.subplots()
        top_careers.plot(kind="bar", color="skyblue", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No logs yet. Try predicting a career first.")
