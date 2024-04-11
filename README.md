 # Docu Buddy

Docu Buddy is an AI-powered chatbot designed to read through documents and answer questions posed by users. Utilizing advanced Natural Language Processing (NLP) techniques, Docu Buddy can understand the context of various documents, making it an invaluable tool for anyone looking to quickly find information within large text files. Built with Python, spaCy, and Hugging Face's Transformers, it's particularly useful for processing technical documents, manuals, and reports.

## Features

- **Document Understanding**: Leverages NLP to understand and process the contents of documents.
- **Efficient Information Retrieval**: Quickly finds and delivers answers from documents.
- **REST API**: Offers a simple REST API for easy integration with web interfaces or other applications.
- **Scalability**: Designed to handle documents up to 500MB efficiently.

## Getting Started

Follow these instructions to get Docu Buddy up and running on your local machine for development and testing purposes.

### Prerequisites

Before you start, ensure you have Python 3.x installed on your system. Then, install the required Python libraries using pip:

```bash
pip install flask spacy transformers
python -m spacy download en_core_web_sm
