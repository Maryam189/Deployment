import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the tokenizer and model from the saved_model folder
model_save_path = './bart_saved_model'  # Ensure this path is correct
tokenizer = BartTokenizer.from_pretrained(model_save_path)
model = BartForConditionalGeneration.from_pretrained(model_save_path)

@app.route('/')
def home():
    return render_template('home.html', input_text='')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    text = request.form['text']
    
    # Generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('home.html', prediction_text=summary, input_text=text)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    text = data['text']
    
    # Generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
