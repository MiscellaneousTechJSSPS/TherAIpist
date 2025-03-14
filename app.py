from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
import os
import re
from model import load_and_preprocess_data, train_model, predict_mental_health
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Flask sessions require a secret key
socketio = SocketIO(app)

# Load dataset and train models
file_path = "data.csv"
df, encoders = load_and_preprocess_data(file_path)

if df is not None:
    depression_model, suicidal_model = train_model(df)
else:
    depression_model, suicidal_model = None, None

# Hugging Face API setup
huggingfacehub_api_token = os.getenv('API_KEY')
huggingface_llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    huggingfacehub_api_token=huggingfacehub_api_token
)

# Define AI prompt template for compassionate responses
prompt_template = PromptTemplate(
    input_variables=["message"],
    template="""
        You are TherAIpy, a compassionate, AI-powered mental wellness companion designed to support emotional health—anytime, anywhere.
        Whether the user is managing stress, navigating anxiety, or just needs someone to talk to, your responses should be thoughtful, caring, and empathetic.
        Blending the power of artificial intelligence with the heart of empathetic care, TherAIpy isn't here to replace real therapy—it's here to walk alongside the user. 
        Be supportive, kind, and encouraging.
        User: {message}
        TherAIpy (as a therapist):"""
)

# Define questionnaire
question_order = [
    "Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure",
    "CGPA", "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
    "Dietary Habits", "Degree", "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness"
]

question_texts = {
    "Gender": "What is your gender? (male/female/other)",
    "Age": "How old are you?",
    "City": "Which city do you live in?",
    "Profession": "What is your profession?",
    "Academic Pressure": "On a scale of 1-5, how much academic pressure do you feel?",
    "Work Pressure": "On a scale of 1-5, how much work pressure do you experience?",
    "CGPA": "What is your CGPA? (Enter a number between 0-10)",
    "Study Satisfaction": "On a scale of 1-5, how satisfied are you with your studies?",
    "Job Satisfaction": "On a scale of 1-5, how satisfied are you with your job?",
    "Sleep Duration": "How long do you sleep? (less than 5 hours / 5-6 hours / 7-8 hours / more than 8 hours)",
    "Dietary Habits": "How would you describe your dietary habits? (healthy/moderate/unhealthy)",
    "Degree": "What is your highest degree?",
    "Work/Study Hours": "How many hours do you work or study daily? (Numeric value)",
    "Financial Stress": "On a scale of 1-5, how much financial stress do you have?",
    "Family History of Mental Illness": "Does your family have a history of mental illness? (yes/no)"
}

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("message")
def handle_message(msg):
    # Check if the conversation is in questionnaire mode
    if msg == "/start":
        session['responses'] = {}  # Reset previous responses
        session['question_index'] = 0
        emit("response", "Hello! Let's assess your mental health. Please answer a few questions honestly.")
        emit("response", question_texts[question_order[0]])  # Ask the first question
        return

    if "question_index" in session and session["question_index"] < len(question_order):
        current_question = question_order[session["question_index"]]
        session['responses'][current_question] = msg  # Store user response
        session["question_index"] += 1

        if session["question_index"] < len(question_order):
            next_question = question_order[session["question_index"]]
            emit("response", question_texts[next_question])  # Ask next question
        else:
            # All questions answered, run prediction
            input_data = session["responses"]
            if depression_model and suicidal_model:
                depression_status, suicidal_status = predict_mental_health(depression_model, suicidal_model, encoders, input_data)
                emit("response", f"Depression: {depression_status}, Suicidal Thoughts: {suicidal_status}")

                # Once mental health predictions are made, pass message to Hugging Face model for compassionate response
                user_message = f"Depression status: {depression_status}, Suicidal Thoughts status: {suicidal_status}. How can I feel better?"
                formatted_prompt = prompt_template.format(message=user_message)
                ai_response = huggingface_llm.invoke(formatted_prompt)
                cleaned_response = re.sub(r"<.*?>", "", ai_response).strip()
                emit("response", cleaned_response)
            else:
                emit("response", "Error: Model not available.")
            del session["question_index"]  # Remove index to allow free chat after questionnaire
        return

    # If outside questionnaire, use AI chatbot
    formatted_prompt = prompt_template.format(message=msg)
    ai_response = huggingface_llm.invoke(formatted_prompt)
    cleaned_response = re.sub(r"<.*?>", "", ai_response).strip()
    emit("response", cleaned_response)

if __name__ == "__main__":
    socketio.run(app, debug=True)