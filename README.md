Description:
Built an AI-powered web application that helps Indian students discover their ideal career paths based on their educational stream, skills, and interests. The app uses a Random Forest Classifier to predict the top 3 career options and integrates with Cohere’s language model to generate a personalized career report. The final PDF-like report is sent directly to the user via SendGrid email integration.

Key Features:

✅ Takes input: Stream (Science/Commerce/Arts), Skills, Interests

🌲 Uses a pre-trained Random Forest ML model for top-3 career prediction

📊 Career labels encoded using LabelEncoder for efficient multi-class prediction

🤖 Sends top career to Cohere’s LLM API to generate detailed, human-like descriptions

🏫 Report includes:

Career Overview

Top Colleges (India-specific)

Average Salary Range

Pros and Cons of the Career

📧 Final report is emailed using SendGrid API with custom formatting

📈 Built-in Analytics Dashboard to visualize past predictions and trends using Matplotlib

Outcome:
✔️ Fully functional, interactive app with real-time career guidance
✔️ Deployed locally via Streamlit, with clean UI and secret management through .toml files
✔️ Emulates a modern, AI-powered career counseling tool
