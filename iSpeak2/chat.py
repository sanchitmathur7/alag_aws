from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('pytorch-chatbot/intents.json', 'r') as json_data:
   intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(sentence):
   sentence = tokenize(sentence)
   X = bag_of_words(sentence, all_words)
   X = X.reshape(1, X.shape[0])
   X = torch.from_numpy(X).to(device)

   output = model(X)
   _, predicted = torch.max(output, dim=1)

   tag = tags[predicted.item()]

   probs = torch.softmax(output, dim=1)
   prob = probs[0][predicted.item()]
   if prob.item() > 0.75:
       for intent in intents['intents']:
           if tag == intent["tag"]:
               return random.choice(intent['responses'])

   return "I do not understand..."

@app.route("/")
def home():
   return render_template("chatbot.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
   user_input = request.form["msg"]
   response = get_response(user_input)
   return jsonify({"response": response})

if __name__ == "__main__":
   app.run(debug=True)