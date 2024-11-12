import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Function to extract text from a PDF file using PyPDF2
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# Streamlit app
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("Automated Text Summarization")

# Sidebar
st.sidebar.title("Menu")
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
input_text = st.sidebar.text_area("Or enter your text here:", height=300)

if pdf_file:
    # Save the uploaded PDF file locally
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf("uploaded_pdf.pdf")
    input_text = st.sidebar.text_area("Extracted text from PDF:", pdf_text, height=300)

if st.sidebar.button("Generate Summary") and (input_text or pdf_file):
    # Combine text from PDF and user input
    combined_text = input_text if input_text else pdf_text

    # Tokenize the input text
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split text into chunks if it exceeds the model's maximum token limit
    max_input_length = 1024  # Max length for BART models
    tokens = tokenizer.encode(combined_text, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Generate summary in chunks if necessary
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summary_text = ""

    # Generate summaries in chunks to avoid exceeding input length
    for i in range(0, tokens.size(1), max_input_length):
        chunk = tokens[:, i:i + max_input_length]
        summary_ids = model.generate(
            chunk,
            max_length=10000,  # Adjusted to allow longer summaries
            min_length=150,  # Adjusted for more comprehensive summaries
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary_text += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "

    # Display the summary in the main content area
    st.subheader("Generated Summary:")
    st.write(summary_text)
