# AI History & Math Teacher

A simple, friendly AI assistant that teaches History and Mathematics using a RAG (Retrieval-Augmented Generation) pipeline.

## Features
- **Role-Playing AI**: Acts strictly as a kind and helpful teacher.
- **RAG Powered**: Uses provided documents (`data/*.txt`) to answer questions accurately.
- **Simple UI**: Clean HTML/CSS interface.
- **Modern Stack**: Built with Python, FastAPI, LangChain, and ChromaDB.

## Setup

1.  **Clone or Download** this project.

2.  **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # OR if that doesn't work:
    python3 -m pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    - Rename `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Open `.env` and add your Groq API Key:
      ```
      GROQ_API_KEY=gsk_your_key_here...
      ```

## Usage

1.  **Run the Application**:
    ```bash
    python main.py
    ```

2.  **Open the Web Interface**:
    Go to [http://localhost:8000](http://localhost:8000) in your browser.

3.  **Ask Questions**:
    Try asking:
    - "Explain the Pythagorean theorem."
    - "What happened during the French Revolution?"
    - "What is Calculus?"

## Customization
- Add more `.txt` files to the `data/` folder to expand the teacher's knowledge base. The app will automatically load them on restart.
