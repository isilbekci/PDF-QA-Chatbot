import os
import openai
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr

openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file given its path.
    Returns the combined text from all pages.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def pdf_extractor(pdf_path):
    """
    Extracts text from the PDF, creates text chunks,
    generates embeddings for these chunks, and indexes them using FAISS.
    Returns the FAISS index, embedding model, text chunks, and the full text.
    """
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text.strip() == "":
        print("No extractable text found in the PDF. Exiting...")
        exit()

    try:
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        exit()

    chunks = pdf_text.split('\n\n')
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    return index, embedding_model, chunks, pdf_text

def get_top_n_answers(pdf_path, question, model, index, chunks, n=3):
    """
    Retrieves the top N most relevant answers from the PDF content
    based on the user's question.
    """
    _, _, _, pdf_text = pdf_extractor(pdf_path)
    question += "this is pdf_text: " + pdf_text
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), n)
    answers = []
    for idx in indices[0]:
        answers.append(chunks[idx])
    return answers

def openai_chatbot(question, pdf_text):
    """
    Generates a detailed response to the user's question using OpenAI GPT-4o.
    """
    question += "this is pdf_text: " + pdf_text
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return e

def chatbot_gradio_interface(file_path, question):
    """
    Integrates the extraction, embedding, and chatbot functions to provide answers
    through a Gradio interface.
    """
    index, embedding_model, chunks, pdf_text = pdf_extractor(file_path)
    context_answers = get_top_n_answers(file_path, question, embedding_model, index, chunks)
    context_response = "\n".join(f"- {answer}" for answer in context_answers)
    detailed_response = openai_chatbot(question, pdf_text)
    return detailed_response

gradio_interface = gr.Interface(
    fn=chatbot_gradio_interface,
    inputs=[gr.File(label = "Please Upload a File"), gr.Textbox()],
    outputs=["text"],
    title="Chatbot",
    description="Ask a question based on the content of a PDF. The chatbot retrieves answers based on the PDF content and provides a detailed response using GPT-4o.",
)

gradio_interface.launch(share=True, debug=True)
