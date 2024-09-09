# PDF-QA-Chatbot

A PDF-based Question-Answering Chatbot built using GPT-4, Sentence Transformers, FAISS, and Gradio. This application extracts text from PDF documents, generates embeddings for the extracted text, and answers user questions based on the document's content. It provides a user-friendly web interface powered by Gradio.

## Features

- Extract text from PDF files.
- Use Sentence Transformers to generate embeddings for chunks of text.
- Utilize FAISS for efficient similarity search.
- Answer questions based on the content of the uploaded PDF using GPT-4.
- User-friendly interface built with Gradio for easy interaction.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/isilbekci/PDF-QA-Chatbot.git
    cd PDF-QA-Chatbot
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

## Usage

Run the application with the following command:

```bash
python app.py
