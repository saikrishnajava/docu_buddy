import os
from flask import Flask, request, jsonify
import spacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import requests
import re
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

# Initialize NLP model and QA pipeline
nlp = spacy.load("en_core_web_lg")
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

app = Flask(__name__)
response_cache = defaultdict(dict)

# Define the directory containing your documents
documents_dir = os.path.join('..', 'data', 'documents')

def extract_keywords(text):
    doc = nlp(text)
    weights = {'NOUN': 2, 'PROPN': 2, 'VERB': 1.5, 'ADJ': 1.2}
    keywords = Counter({token.lemma_.lower(): weights.get(token.pos_, 1)
                        for token in doc if token.is_alpha and not token.is_stop})
    return keywords.keys()

def fetch_from_wikipedia(question):
    if 'wikipedia' in response_cache[question]:
        return response_cache[question]['wikipedia']
    
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={question}&format=json"
    response = requests.get(search_url)
    if response.status_code != 200:
        return "Failed to fetch data", 0
    
    search_results = response.json().get("query", {}).get("search", [])
    if not search_results:
        return "No information found", 0
    
    title = search_results[0]['title'].replace(' ', '_')
    page_url = f"https://en.wikipedia.org/wiki/{title}"
    
    page_response = requests.get(page_url)
    if page_response.status_code != 200:
        return "Failed to fetch page content", 0
    
    soup = BeautifulSoup(page_response.content, 'html.parser')
    paragraphs = [p.getText() for p in soup.find_all('p') if len(p.getText()) > 100]
    summary = ' '.join(paragraphs[:3])
    
    # Convert dict_keys to lists
    question_keywords = list(extract_keywords(question))
    summary_keywords = list(extract_keywords(summary))
    
    score = sum(question_keywords.count(keyword) * summary_keywords.count(keyword) for keyword in question_keywords)
    
    response_cache[question]['wikipedia'] = (summary, score)
    return summary, score
    

def fetch_from_duckduckgo(question):
    if 'duckduckgo' in response_cache[question]:
        return response_cache[question]['duckduckgo']
    
    search_url = f"https://api.duckduckgo.com/?q={question}&format=json"
    response = requests.get(search_url)
    if response.status_code != 200:
        return "Failed to fetch data", 0
    
    search_results = response.json().get("AbstractText", "")
    if not search_results:
        return "No information found", 0
    
    # Clean the response text
    summary = clean_text(search_results)
    
    # Convert dict_keys to lists
    question_keywords = list(extract_keywords(question))
    summary_keywords = list(extract_keywords(summary))
    
    score = sum(question_keywords.count(keyword) * summary_keywords.count(keyword) for keyword in question_keywords)
    
    response_cache[question]['duckduckgo'] = (summary, score)
    return summary, score

def clean_text(text):
    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    # Remove unwanted characters
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Trim leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def preprocess_documents(documents_dir):
    documents_content = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents_content.append(file.read())
    return documents_content

documents_content = preprocess_documents(documents_dir)

def match_question_to_documents(question_keywords):
    matched_documents = []
    for keyword in question_keywords:
        for filename in os.listdir(documents_dir):
            if keyword in filename.lower():
                matched_documents.append(filename)
    return matched_documents

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    question_keywords = extract_keywords(question)
    matched_documents = match_question_to_documents(question_keywords)
    
    if matched_documents:
        matched_content = [documents_content[matched_documents.index(matched_doc)] for matched_doc in matched_documents]
        matched_content = " ".join(matched_content)
    else:
        matched_content = " ".join(documents_content)

    doc_answer = qa_pipeline(question=question, context=matched_content)
    response_cache[question]['document'] = doc_answer  # Cache the document answer

    #wiki_summary, wiki_score = fetch_from_wikipedia(question)
    duckGo_summary, duckGo_score = fetch_from_duckduckgo(question)

    if doc_answer.get('score', 0) >= 0.05 and doc_answer.get('score', 0) > duckGo_score:
        final_answer = doc_answer['answer']
    else:
        final_answer = duckGo_summary if duckGo_summary else "I'm sorry, I couldn't find any relevant information to answer your question."

    return jsonify({"answer": final_answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
