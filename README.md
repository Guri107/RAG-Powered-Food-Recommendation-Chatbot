# Food Recommendation System

A Python-based food recommendation project that combines vector search with a generative AI chatbot.

## What it does

- Loads food data from `FoodDataSet.json`
- Builds a ChromaDB vector collection using sentence-transformers embeddings
- Supports:
  - `interactive_search.py` for basic similarity search
  - `advanced_search.py` for filtered, calorie, and cuisine-focused search
  - `enhanced_rag_chatbot.py` for RAG-style recommendation using Google Gemini

## Setup

1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create `.env` from the example:
   ```powershell
   copy .env.example .env
   ```

4. Update `.env` with your Gemini API values.

## Running

- Basic search:
  ```powershell
  python interactive_search.py
  ```

- Advanced filtered search:
  ```powershell
  python advanced_search.py
  ```

- Enhanced RAG chatbot:
  ```powershell
  python enhanced_rag_chatbot.py
  ```

## Environment config

The chatbot uses these `.env` keys:

- `GEMINI_API_KEY`
- `GEMINI_PROJECT_NAME`
- `GEMINI_MODEL`
- `GEMINI_URL`

Do not commit `.env` to git.

## Debugging

- Confirm active Python interpreter:
  ```powershell
  python -c "import sys; print(sys.executable)"
  ```

- Check installed packages:
  ```powershell
  python -m pip show chromadb sentence-transformers
  ```

- Run the chatbot in debug mode:
  ```powershell
  python -m pdb enhanced_rag_chatbot.py
  ```

- Validate env vars:
  ```powershell
  python -c "import os; print(os.getenv('GEMINI_API_KEY')); print(os.getenv('GEMINI_URL'))"
  ```

## Notes

- `shared_functions.py` contains the vector store and search helpers.
- `enhanced_rag_chatbot.py` loads Gemini config from `.env` and uses the Google Generative Language API.
- `.gitignore` excludes the virtual env, `.env`, and cache files.
