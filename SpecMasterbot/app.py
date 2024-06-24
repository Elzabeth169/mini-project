from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import spacy
import pandas as pd
import re
from enum import Enum
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
CORS(app)

# Load the intent recognition model
intent_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Load spaCy model for entity extraction
nlp = spacy.load('en_core_web_sm')

# Load the conversational model for general conversations
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
conversational_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load the semantic search model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset with different encodings if needed
try:
    laptops_df = pd.read_csv('complete_laptop_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        laptops_df = pd.read_csv('complete_laptop_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        laptops_df = pd.read_csv('complete_laptop_data.csv', encoding='latin1')

# Strip leading/trailing spaces from column names
laptops_df.columns = laptops_df.columns.str.strip()

# Ensure columns exist in the dataset
required_columns = [
    'Dedicated Graphic Memory Capacity', 'Number of Cores', 'RAM', 'Weight',
    'Screen Size', 'Price', 'name', 'Manufacturer', 'link', 'Operating System', 'Suitable For', 'Processor Name'
]

for col in required_columns:
    if col not in laptops_df.columns:
        laptops_df[col] = None

# Assign purposes to laptops based on specifications
def assign_purpose(row):
    gpu = row['Dedicated Graphic Memory Capacity']
    cores = row['Number of Cores']
    ram = row['RAM']
    weight = row['Weight']
    screen_size = row['Screen Size']

    try:
        gpu = float(re.search(r'\d+', str(gpu)).group()) if pd.notna(gpu) else 0
    except (ValueError, AttributeError):
        gpu = 0

    try:
        cores = int(re.search(r'\d+', str(cores)).group()) if pd.notna(cores) else 0
    except (ValueError, AttributeError):
        cores = 0

    try:
        ram = int(re.search(r'\d+', str(ram)).group()) if pd.notna(ram) else 0
    except (ValueError, AttributeError):
        ram = 0

    try:
        weight = float(re.search(r'\d+(\.\d+)?', str(weight)).group()) if pd.notna(weight) else float('inf')
    except (ValueError, AttributeError):
        weight = float('inf')

    try:
        screen_size = float(re.search(r'\d+(\.\d+)?', str(screen_size)).group()) if pd.notna(screen_size) else float('inf')
    except (ValueError, AttributeError):
        screen_size = float('inf')

    if gpu > 2 or cores >= 8:
        return 'gaming'
    elif ram >= 16 or cores >= 8:
        return 'programming'
    elif weight < 1.5 and screen_size < 35:
        return 'school/work'
    else:
        return 'general use'

laptops_df['purpose'] = laptops_df.apply(assign_purpose, axis=1)
laptops = laptops_df.to_dict(orient='records')

# Ensure each laptop entry has a 'features' key
for laptop in laptops:
    if 'features' not in laptop:
        laptop['features'] = ""

# Define intents
intents = ["budget", "purpose", "brand", "features", "general conversation"]

# Helper function to extract entities
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"budget": None, "purpose": None, "brand": None, "features": None}

    # Extract monetary values for budget
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["budget"] = ent.text
        elif ent.label_ == "ORG":
            entities["brand"] = ent.text

    # Extract purpose and features based on keywords
    purpose_keywords = ['gaming', 'programming', 'school', 'work', 'general']
    feature_keywords = ['16GB RAM', '8GB RAM', 'graphic card', 'SSD', 'lightweight']

    for keyword in purpose_keywords:
        if keyword in user_input.lower():
            entities["purpose"] = keyword

    for keyword in feature_keywords:
        if keyword.lower() in user_input.lower():
            if entities["features"]:
                entities["features"] += f", {keyword}"
            else:
                entities["features"] = keyword

    return entities

# Define the conversation state
class ConversationState(Enum):
    INITIAL = 1
    DETECT_INTENT = 2
    ASK_DETAILS = 3
    ASK_BUDGET = 4
    ASK_PURPOSE = 5
    ASK_BRAND = 6
    ASK_FEATURES = 7
    CONFIRMATION = 8
    PROVIDE_RECOMMENDATION = 9
    GENERAL_CONVERSATION = 10

user_contexts = {}

# Function to generate embeddings for laptops
def generate_embeddings(laptops):
    descriptions = [
        f"{laptop['name']} {laptop['Processor Name']} - {laptop['RAM']} - {laptop['purpose']} - {laptop['Price']} - {laptop['Operating System']}"
        for laptop in laptops
    ]
    return semantic_model.encode(descriptions, convert_to_tensor=True)

laptop_embeddings = generate_embeddings(laptops)

# Function to recommend a laptop using semantic search
def recommend_laptop(context):
    budget = context.get("budget")
    purpose = context.get("purpose")
    brand = context.get("brand")
    features = context.get("features")

    def match_features(laptop, features):
        if not features:
            return True
        feature_list = features.lower().split(',')
        for feature in feature_list:
            if feature.strip() not in laptop.get('features', '').lower():
                return False
        return True

    query = f"laptop with a budget of {budget or 'any'}, for {purpose or 'any'} use, preferring {brand or 'any'} brand, and with features like {features or 'any'}"
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)

    # Calculate similarity scores
    cos_scores = util.pytorch_cos_sim(query_embedding, laptop_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    matching_laptops = [laptops[i] for i in top_results[1]]

    if matching_laptops:
        response = "Here are some laptops I recommend:\n"
        i = 1
        for laptop in matching_laptops:
            response += f"{i}. {laptop['name']} {laptop['Processor Name']} - {laptop['RAM']} - {laptop['purpose']} - {laptop['Price']} - {laptop['Operating System']} - {laptop['link']}\n"
            i += 1
        return response
    else:
        return "I couldn't find any laptops matching your criteria."

@app.route('/')
def index():
    return render_template('bot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_id = request.json.get('user_id')
    user_input = request.json.get('message')

    # Initialize response_text to avoid UnboundLocalError
    response_text = ""

    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "state": ConversationState.INITIAL,
            "budget": None,
            "purpose": None,
            "brand": None,
            "features": None,
            "conversation_history": None  # Initialize conversation history as None
        }

    context = user_contexts[user_id]
    state = context["state"]

    if state == ConversationState.INITIAL:
        context["state"] = ConversationState.DETECT_INTENT
        response_text = "How can I assist you today?"

    elif state == ConversationState.DETECT_INTENT:
        intents_result = intent_classifier(user_input, intents)
        intent = intents_result['labels'][0]
        if intent == "general conversation":
            context["state"] = ConversationState.GENERAL_CONVERSATION
            response_text = "Sure, let's have a chat. What's on your mind?"
        else:
            entities = extract_entities(user_input)
            context.update(entities)
            context["state"] = ConversationState.CONFIRMATION
            response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.ASK_DETAILS:
        entities = extract_entities(user_input)
        context.update(entities)
        if not context["budget"]:
            context["state"] = ConversationState.ASK_BUDGET
            response_text = "What is your budget for the laptop?"
        elif not context["purpose"]:
            context["state"] = ConversationState.ASK_PURPOSE
            response_text = "What will you be using the laptop for? (e.g., gaming, programming, school/work, general use)"
        elif not context["brand"]:
            context["state"] = ConversationState.ASK_BRAND
            response_text = "Do you have any preferred brands?"
        elif not context["features"]:
            context["state"] = ConversationState.ASK_FEATURES
            response_text = "Any specific features you're looking for?"
        else:
            context["state"] = ConversationState.CONFIRMATION
            response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.ASK_BUDGET:
        context["budget"] = user_input
        if not context["purpose"]:
            context["state"] = ConversationState.ASK_PURPOSE
            response_text = "What will you be using the laptop for? (e.g., gaming, programming, school/work, general use)"
        elif not context["brand"]:
            context["state"] = ConversationState.ASK_BRAND
            response_text = "Do you have any preferred brands?"
        elif not context["features"]:
            context["state"] = ConversationState.ASK_FEATURES
            response_text = "Any specific features you're looking for?"
        else:
            context["state"] = ConversationState.CONFIRMATION
            response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.ASK_PURPOSE:
        context["purpose"] = user_input
        if not context["brand"]:
            context["state"] = ConversationState.ASK_BRAND
            response_text = "Do you have any preferred brands?"
        elif not context["features"]:
            context["state"] = ConversationState.ASK_FEATURES
            response_text = "Any specific features you're looking for?"
        else:
            context["state"] = ConversationState.CONFIRMATION
            response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.ASK_BRAND:
        context["brand"] = user_input
        if not context["features"]:
            context["state"] = ConversationState.ASK_FEATURES
            response_text = "Any specific features you're looking for?"
        else:
            context["state"] = ConversationState.CONFIRMATION
            response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.ASK_FEATURES:
        context["features"] = user_input
        context["state"] = ConversationState.CONFIRMATION
        response_text = f"Got it. Just to confirm, you want a laptop with a budget of {context['budget'] or 'not specified'}, for {context['purpose'] or 'not specified'} use, preferring {context['brand'] or 'any'} brand, and with features like {context['features'] or 'no specific features'}. Is that correct? (yes/no)"

    elif state == ConversationState.CONFIRMATION:
        if user_input.lower() in ["yes", "y"]:
            context["state"] = ConversationState.PROVIDE_RECOMMENDATION
            response_text = recommend_laptop(context)
        else:
            context["state"] = ConversationState.ASK_DETAILS
            response_text = "Let's start over. What is your budget for the laptop?"

    elif state == ConversationState.PROVIDE_RECOMMENDATION:
        context["state"] = ConversationState.INITIAL
        response_text = "Is there anything else I can assist you with?"

    elif state == ConversationState.GENERAL_CONVERSATION:
        if context.get("conversation_history") is None:
            context["conversation_history"] = user_input
        else:
            context["conversation_history"] += tokenizer.eos_token + user_input

        # Encode input and generate response
        new_user_input_ids = tokenizer.encode(context["conversation_history"] + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = conversational_model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        context["conversation_history"] += tokenizer.eos_token + response_text

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)


