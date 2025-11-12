# Local-RAG-with-NorMistral-7B-Warm-Instruct


This project implements a fully local Retrieval-Augmented Generation (RAG) system using the NorMistral-7B-Warm-Instruct model served through llama.cpp.

You can ask natural-language questions about your own research papers stored as PDFs, and the system will provide concise, grounded answers drawn directly from those documents — all offline and without any cloud APIs.

System Architecture

PDF Files (research papers)
       │
       ▼
SentenceTransformer (embeddings)
       │
       ▼
Retriever (semantic search)
       │
       ▼
ask_nor_model()  ──►  llama-server (NorMistral-7B-Warm-Instruct)
                            │
                            ▼
                    Context-grounded Answer


Requirements

| Component | Specification                             |
| --------- | ----------------------------------------- |
| OS        | Windows 10/11 (x64)                       |
| RAM       | ≥ 16 GB                                   |
| GPU       | NVIDIA RTX 3050 (4 GB VRAM)               |
| Python    | 3.10 (recommended)                        |
| Model     | `normistral-7b-warm-instruct.Q3_K_M.gguf` |
| Backend   | `llama-server.exe` from **llama.cpp**     |


Folder Structure

C:\
 └── llama_cpp\
      ├── llama-server.exe
      ├── llama.dll
      ├── models\
      │    └── normistral-7b-warm-instruct.Q3_K_M.gguf
      └── papers\
           ├── paper1.pdf
           ├── paper2.pdf
           └── paper3.pdf


Downloads

| Resource                               | Source                                       | Link                                                                                                                     |
| -------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **NorMistral-7B-Warm-Instruct (GGUF)** | NORA LLM / University of Oslo @ Hugging Face | [https://huggingface.co/norallm/normistral-7b-warm-instruct](https://huggingface.co/norallm/normistral-7b-warm-instruct) |
| **llama.cpp Prebuilt (Windows)**       | Georgi Gerganov / llama.cpp GitHub Releases  | [https://github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases)                       |
| **Example CUDA Build**                 | For NVIDIA GPUs                              | e.g. `llama-b<version>-bin-win-cu12-x64.zip`                                                                             |


Running the Model Server

Open Command Prompt and execute:

cd C:\llama_cpp
llama-server.exe -m "C:\llama_cpp\models\normistral-7b-warm-instruct.Q3_K_M.gguf" -c 1024 --host 127.0.0.1 --port 8080


Optional (CUDA acceleration):

llama-server.exe -m "C:\llama_cpp\models\normistral-7b-warm-instruct.Q3_K_M.gguf" -c 1024 --host 127.0.0.1 --port 8080 --gpu-layers 20


When you see:

main: server is listening on http://127.0.0.1:8080

the API is ready.


🧮 Notebook Usage

All processing, retrieval, and model interaction logic resides in the uploaded Jupyter notebook (main.ipynb).
The notebook:
Reads all PDFs from C:\llama_cpp\papers\
Extracts and chunks text for embedding
Computes semantic embeddings using sentence-transformers
Performs vector similarity search for each user query
Sends retrieved context and the question to the local NorMistral model
Displays a context-grounded answer

🧠 Why llama-server is Needed

llama-server is the inference backend that:
Loads the quantized NorMistral model into memory
Generates responses locally on your CPU/GPU
Exposes a REST API (/v1/chat/completions) that the notebook calls via HTTP
It acts as your local equivalent of OpenAI’s API, allowing Python to communicate with a fully local LLM.

🧠 How It Works

Start the llama-server to host the NorMistral model.
Launch the notebook and run all cells in order.
The notebook retrieves relevant content from your PDFs and queries the local model.
The model replies using only the information retrieved from your documents.

💡 Notes

You can raise the context window if needed: -c 2048.
Larger quantizations (Q4/Q5) improve quality but require more RAM.
To run CPU-only, omit --gpu-layers.
All computation and data remain local — no internet required.

🧾 License & Credits

NorMistral-7B-Warm-Instruct © NORA LLM / University of Oslo — Hugging Face Model Card
Project created for educational and research use.
