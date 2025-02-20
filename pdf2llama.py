import os
import time
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaLLM

def extract_text_with_langchain_pdf(pdf_file: str) -> str:
    """Extracts text from a PDF using LangChain, removing page numbers and footers."""
    print(f"Extracting text from {pdf_file}...")
    documents = UnstructuredLoader(pdf_file).load()
    pdf_pages_content = ""
    
    for doc in documents:
        page_content = doc.page_content
        page_content = re.sub(r"\b\d+\b(?=\s*$)", "", page_content, flags=re.MULTILINE)
        page_content = re.sub(r"Footer text pattern.*$", "", page_content, flags=re.MULTILINE)
        pdf_pages_content += page_content + '\n'
    
    return pdf_pages_content

def generate_question_answer_pair(chunk, chunk_number, total_chunks, ollama_llama32, ollama_mistral):
    """Generates a question and answer pair concurrently using two different models."""
    print(f"Processing chunk {chunk_number}/{total_chunks}...")
    

        #Adjust the question and answer prompt according to your specific needs and the role or task aligned with your LLM training focus.
        #For example, a research assistance prompt template could be:
        #“I’m conducting a study on [Topic/Subject]. Could you provide relevant research articles, resources, or expert opinions to support my work?”  
    def get_question():
        return ollama_llama32.invoke(f"You are an experienced automotive software engineer. Read the following text and create a question related to functional safety:\n\n{chunk}\n\nQuestion:").strip()
    def get_answer(question):
        return ollama_mistral.invoke(f"You are an experienced automotive software engineer. Read the following text and answer the question:\n\nText:\n{chunk}\n\nQuestion:\n{question}\n\nAnswer:").strip()
    
    with ThreadPoolExecutor() as executor:
        future_question = executor.submit(get_question)
        question = future_question.result()
        future_answer = executor.submit(get_answer, question)
        answer = future_answer.result()
    
    return {"instruction": question, "input": chunk, "output": answer}

def generate_qa_json(pdf_file: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """Reads a PDF, chunks the content, and generates QA pairs using Ollama in parallel."""
    print(f"Starting QA generation for: {pdf_file}")
    start_time = time.time()
    text = extract_text_with_langchain_pdf(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    
    # Initialize two instances of the OllamaLLM class with different models:
    # 1. `ollama_llama32` uses the "llama3.2" model with settings:
    #    - top_p=0.85: Controls how much randomness is allowed in the results
    #    - temperature=0.3: Makes the output more predictable (lower = more predictable)
    #    - top_k=50: Limits the next word choices to the top 50 options
    # 2. `ollama_mistral` uses the "mistral" model with the same settings.
    # You can change the model or settings if needed.
    # Keep the temprature low if you'r working on technical things :)

    ollama_llama32 = OllamaLLM(model="llama3.2", model_kwargs={"top_p": 0.85, "temperature": 0.3, "top_k": 50})
    ollama_mistral = OllamaLLM(model="mistral", model_kwargs={"top_p": 0.85, "temperature": 0.3, "top_k": 50})

    qa_pairs = []
    total_chunks = len(texts)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_question_answer_pair, chunk, i+1, total_chunks, ollama_llama32, ollama_mistral) for i, chunk in enumerate(texts)]
        for future in futures:
            qa_pairs.append(future.result())
    
    elapsed_time = time.time() - start_time
    print(f"QA generation completed for {pdf_file} in {elapsed_time:.2f} seconds")
    return qa_pairs, elapsed_time

def generate_unique_filename(base_name):
    """Generates a unique filename if the file already exists."""
    counter = 1
    file_name = f"qa_data_{base_name}.json"
    while os.path.exists(file_name):
        file_name = f"qa_data_{base_name}_{counter}.json"
        counter += 1
    return file_name

def process_pdf(pdf_path: Path):
    """Processes a single PDF file."""
    pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
    base_name = pdf_path.stem
    output_file = generate_unique_filename(base_name)
    
    qa_data, processing_time = generate_qa_json(str(pdf_path))
    
    with open(output_file, "w") as f:
        json.dump(qa_data, f, indent=4)
    
    json_size = os.path.getsize(output_file) / 1024  # Size in KB
    summary = {
        "pdf_file": str(pdf_path),
        "pdf_size_MB": round(pdf_size, 2),
        "json_file": output_file,
        "json_size_KB": round(json_size, 2),
        "qa_pairs": len(qa_data),
        "processing_time_sec": round(processing_time, 2)
    }

    print(f"Processed {pdf_path}:")
    for key, value in summary.items():
            print(f"  - {key.replace('_', ' ').capitalize()}: {value}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Process a single PDF or all PDFs in a folder.")
    parser.add_argument("path", type=str, help="Path to a PDF file or folder containing PDFs")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_dir():
        print(f"Processing all PDFs in folder: {path}")
        pdf_files = list(path.glob("*.pdf"))
    elif path.is_file() and path.suffix.lower() == ".pdf":
        pdf_files = [path]
    else:
        print("Invalid input. Provide a valid PDF file or folder containing PDFs.")
        return
    
    summaries = []
    for pdf in pdf_files:
        summaries.append(process_pdf(pdf))
    
    # Save processing summary
    with open("processing_summary.json", "w") as f:
        json.dump(summaries, f, indent=4)
    print("Processing completed. Summary saved in processing_summary.json")

if __name__ == "__main__":
    main()
