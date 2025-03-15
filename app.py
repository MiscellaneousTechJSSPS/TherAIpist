import re
from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
import os
from model import load_and_preprocess_data, train_model, predict_mental_health
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["API_KEY"] = os.getenv('API_KEY')
socketio = SocketIO(app)

file_path = "data.csv"
df, encoders = load_and_preprocess_data(file_path)

if df is not None:
    depression_model, suicidal_model = train_model(df)
else:
    depression_model, suicidal_model = None, None

huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_KEY"]
huggingface_llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    huggingfacehub_api_token=huggingfacehub_api_token
)

prompt_template = PromptTemplate(
    input_variables=["message"],
    template="""You are TherAIpist, a compassionate, AI-powered mental wellness companion designed to support emotional health—anytime, anywhere.
                Whether the user is managing stress, navigating anxiety, or just needs someone to talk to, your responses should be thoughtful, caring, and empathetic.
                Blending the power of artificial intelligence with the heart of empathetic care, TherAIpist isn't here to replace real therapy—it's here to walk alongside the user. 
                Be supportive, kind, and encouraging.
                Ask them about their hobbies or favourite movies or activities and recommend helpful resources or coping strategies like relatable movies, live matches online or live matches available to play around them, trips and places around them, etc.
                DO NOT HELP WITH ANYTHING UNRELATED TO MENTAL HEALTH!!
                Also, if they ask, please recommend a professional therapist around them.
                DO NOT REPEAT YOURSELF.
                User: {message}
                TherAIpist (as a therapist):"""
)

def remove_repeated_text(response):
    # Strip extra spaces and remove repeated sentences or words
    response = " ".join(response.split())
    pattern = re.compile(r"(.*?)(\s*\1)+", re.DOTALL)
    response = pattern.sub(r"\1", response).strip()
    return response

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
    if msg == "/start":
        session['responses'] = {}
        session['question_index'] = 0
        emit("response", "Hello! Let's assess your mental health. Please answer a few questions honestly.")
        emit("response", question_texts[question_order[0]])
        return

    if "question_index" in session and session["question_index"] < len(question_order):
        current_question = question_order[session["question_index"]]
        session['responses'][current_question] = msg
        session["question_index"] += 1

        if session["question_index"] < len(question_order):
            next_question = question_order[session["question_index"]]
            emit("response", question_texts[next_question])
        else:
            input_data = session["responses"]
            if depression_model and suicidal_model:
                depression_status, suicidal_status = predict_mental_health(depression_model, suicidal_model, encoders, input_data)
                emit("response", f"Depression: {depression_status}, Suicidal Thoughts: {suicidal_status}")

                user_message = f"Depression status: {depression_status}, Suicidal Thoughts status: {suicidal_status}. How can I feel better?"
                formatted_prompt = prompt_template.format(message=user_message)
                ai_response = huggingface_llm.invoke(formatted_prompt)
                cleaned_response = re.sub(r"<.*?>", "", ai_response).strip()
                cleaned_response = remove_repeated_text(cleaned_response)
                emit("response", cleaned_response)
            else:
                emit("response", "Error: Model not available.")
            del session["question_index"]
        return

    formatted_prompt = prompt_template.format(message=msg)
    ai_response = huggingface_llm.invoke(formatted_prompt)
    cleaned_response = re.sub(r"<.*?>", "", ai_response).strip()
    cleaned_response = remove_repeated_text(cleaned_response)
    emit("response", cleaned_response)

if __name__ == "__main__":
    socketio.run(app, debug=True)