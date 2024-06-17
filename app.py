from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS
from enum import Enum
import re  # Import the 're' module
import random  # Import the 'random' module

app = Flask(__name__)
CORS(app)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

class ConversationState(Enum):
    INITIAL = 1
    ASK_BUDGET = 2
    ASK_PURPOSE = 3
    ASK_BRAND = 4
    ASK_FEATURES = 5
    PROVIDE_RECOMMENDATION = 6

user_contexts = {}

laptops = [
    {'name': 'Acer 4', 'ram': '8GB', 'purpose': 'general use', 'price': 978, 'brand': 'Acer', 'features': 'SSD'},
    {'name': 'Dell 3', 'ram': '4GB', 'purpose': 'general use', 'price': 634, 'brand': 'Dell', 'features': 'HDD'},
    {'name': 'Dell 3', 'ram': '8GB', 'purpose': 'general use', 'price': 946, 'brand': 'Dell', 'features': 'SSD'},
    {'name': 'Dell 4', 'ram': '8GB', 'purpose': 'gaming', 'price': 1244, 'brand': 'Dell', 'features': 'SSD'},
    {'name': 'HP 4', 'ram': '8GB', 'purpose': 'general use', 'price': 837, 'brand': 'HP', 'features': 'SSD'}
]

def extract_info(user_input):
    tokens = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(tokens, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def recommend_laptop(context):
    budget = context.get("budget")
    purpose = context.get("purpose")
    brand = context.get("brand")
    features = context.get("features")

    # Filter laptops based on user criteria
    filtered_laptops = [laptop for laptop in laptops if 
                        (budget is None or laptop["price"] <= int(budget)) and 
                        (purpose is None or purpose.lower() in laptop["purpose"].lower()) and
                        (brand is None or brand.lower() in laptop["brand"].lower()) and
                        (features is None or features.lower() in laptop["features"].lower())]

    if filtered_laptops:
        # Sort laptops by the number of matching criteria
        filtered_laptops.sort(key=lambda x: sum([
            budget is not None and x["price"] <= int(budget),
            purpose is not None and purpose.lower() in x["purpose"].lower(),
            brand is not None and brand.lower() in x["brand"].lower(),
            features is not None and features.lower() in x["features"].lower()
        ]), reverse=True)

        response = "Here are some laptops I recommend:\n"
        for laptop in filtered_laptops[:3]:  # Recommend up to 3 laptops
            response += f"{laptop['name']} - {laptop['ram']} - {laptop['purpose']} - ${laptop['price']} - {laptop['brand']} - {laptop['features']}\n"
        return response
    else:
        # Fallback to recommending any laptop if no perfect match is found
        random_laptop = laptops[random.randint(0, len(laptops) - 1)]
        response = f"I couldn't find any laptops matching all your criteria. Here's a random recommendation:\n"
        response += f"{random_laptop['name']} - {random_laptop['ram']} - {random_laptop['purpose']} - ${random_laptop['price']} - {random_laptop['brand']} - {random_laptop['features']}"
        return response

@app.route('/')
def index():
    return render_template('bot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_id = request.json.get('user_id')
    user_input = request.json.get('message')
    
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "state": ConversationState.INITIAL,
            "budget": None,
            "purpose": None,
            "brand": None,
            "features": None
        }

    context = user_contexts[user_id]
    state = context["state"]

    if state == ConversationState.INITIAL:
        context["state"] = ConversationState.ASK_BUDGET
        response_text = "What is your budget for the laptop?"

    elif state == ConversationState.ASK_BUDGET:
        match = re.search(r"\d+", user_input)
        if match:
            context["budget"] = match.group()
            context["state"] = ConversationState.ASK_PURPOSE
            response_text = "What will you be using the laptop for? (e.g., gaming, programming, general use)"
        else:
            response_text = "I couldn't find a budget in your response. Could you please provide your budget in numerical form?"

    elif state == ConversationState.ASK_PURPOSE:
        context["purpose"] = user_input
        context["state"] = ConversationState.ASK_BRAND
        response_text = "Do you have any brand preferences? (e.g., Dell, HP, Apple)"

    elif state == ConversationState.ASK_BRAND:
        context["brand"] = user_input
        context["state"] = ConversationState.ASK_FEATURES
        response_text = "Are there any specific features you need? (e.g., 16GB RAM, SSD storage)"

    elif state == ConversationState.ASK_FEATURES:
        context["features"] = user_input
        context["state"] = ConversationState.PROVIDE_RECOMMENDATION
        response_text = recommend_laptop(context)
        # Reset context for the user after providing a recommendation
        user_contexts[user_id] = {
            "state": ConversationState.INITIAL,
            "budget": None,
            "purpose": None,
            "brand": None,
            "features": None
        }

    else:
        response_text = "Sorry, I didn't understand that."

    # Print the response for debugging
    print(f"User input: {user_input}")
    print(f"Bot response: {response_text}")

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)

