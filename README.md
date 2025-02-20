# pdf2llama QA Data Generator

## Overview
This project processes PDF documents, extracts text, and generates Question-Answer (QA) pairs using two LLMs (Large Language Models). One LLM generates questions from extracted text chunks, while another provides answers based on its understanding. The results are saved in a meta prompt structured JSON format.
Built with **LangChain**, it leverages **multi-threaded processing** and **GPU acceleration** to efficiently generate training-ready data.🤖📚

---

## Features
- **Automated PDF parsing** with text extraction
- **Chunk-based processing** for better context handling
- **Multi-threaded question generation** (LLM1) and answer generation (LLM2)
- **JSON output in a structured format**, ready for LLM training
- **Supports GPU acceleration** (NVIDIA CUDA) but also runs on CPU
- **Cross-platform support**: **Ubuntu, Mac, and Windows**

---

### Prerequisites📦
Ensure you have the following installed:
- Python 3.8+
- pip
- NVIDIA CUDA (for GPU acceleration, optional)

### Install Dependencies
```bash
pip install -r requirements.txt

pip install langchain langchain_unstructured langchain_ollama
```
- Standard libraries used: `os`, `time`, `json`, `re`, `argparse`, `concurrent.futures`, `pathlib`, `typing`.

---

## Configuration ⚙️

- Ensure Ollama API is accessible, and models (`llama3.2` and `mistral`) are configured.
- Modify model parameters (e.g., `top_p`, `temperature`, `top_k`) if needed.
- Provide a valid path to a PDF or a folder containing PDFs.

---

## How It Works 🛠️

1. **Extracts and Cleans PDF Text** using LangChain.
2. **Splits Text** into smaller chunks.
3. **Generates QA Pairs Concurrently**:
   - **LLM1 (Question Generator)**: Generates questions from text chunks
   - `llama3.2` creates questions.
   - **LLM2 (Answer Generator)**: Answers the generated questions
   - `mistral` generates answers.
4. **Output Processing**
   - Aggregates QA pairs into structured JSON format

--

## Usage 🚀

Hint: ⏳ A 450 KB PDF takes approximately 17 minutes to process on a Mac M2 chip. 
Processing time may vary based on hardware and PDF complexity. 🚀

### Process a Single PDF:

```bash
python pdf2llama.py path/to/yourfile.pdf
```

---

### Process All PDFs in a Folder:

```bash
python pdf2llama.py path/to/your/folder
```

JSON files with QA pairs and summaries will be generated.

---

## Architecture Overview
!grafik[Uploading PDF2Llama.jpg…]()

--

## JSON Output Format
```json
[
  {
    "instruction": "Extracted text from the PDF",
    "question": "Generated question",
    "answer": "Generated answer"
  }
]
```

## GPU Optimization
To enable GPU acceleration:
```bash
export CUDA_VISIBLE_DEVICES=0
```

For CPU-only execution:
```bash
python pdf2llama.py --use_cpu
```

## Running Environments 🌐

- **Local Machines**: Windows, macOS, Linux.
- **Cloud Environments**: AWS, Google Cloud, Azure.
- **Docker**: Easily containerized.

## Contributing 🤝
Contributions are welcome! Fork, create a branch, and submit a pull request.

---

## License 📄

This project is licensed under the MIT License.

---

## Contact 📫
For questions, reach out via GitHub Issues or discussions.

---

Enjoy using PDF2llama! 🎉
