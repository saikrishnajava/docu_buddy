from flask import Flask, request, jsonify
import spacy
from transformers import pipeline
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering")

# Create a Flask app
app = Flask(__name__)

# Function to read and preprocess the documents
def preprocess_documents(documents_dir):
    documents_content = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents_content.extend(file.read().split('\n\n'))
    return documents_content

# Preprocess and store the documents' contents
documents_dir = os.path.join('..', 'data', 'documents')  # Adjust the path to where you store your documents
documents_content = preprocess_documents(documents_dir)

@app.route('/ask', methods=['POST'])
def ask_question():
    # Extract the question from the incoming JSON request
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Find the most relevant section from the preprocessed documents
    relevant_context = find_relevant_context(documents_content, question)
    
    # Get the answer from the QA pipeline
    answer = get_answer(relevant_context, question)
    
    # Return the answer in JSON format
    return jsonify({"answer": answer})

def find_relevant_context(documents_content, question):
    # Extract keywords from the question
    question_doc = nlp(question)
    keywords = [token.lemma_.lower() for token in question_doc if token.pos_ in ("NOUN", "PROPN", "VERB")]

    # Find the most relevant paragraph
    relevant_paragraphs = sorted(
        documents_content,
        key=lambda p: sum(p.lower().count(keyword) for keyword in keywords),
        reverse=True
    )
    return relevant_paragraphs[0] if relevant_paragraphs else ""

def get_answer(context, question):
    # Use the QA pipeline to get an answer
    if context:
        return qa_pipeline(question=question, context=context)['answer']
    else:
        return "I'm sorry, I couldn't find any relevant information to answer your question."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
