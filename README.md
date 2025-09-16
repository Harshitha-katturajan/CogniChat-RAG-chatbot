### CogniChat: Your RAG Assistant

This project is a simple, beginner-friendly chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on specific documents. It's built with **Streamlit** for the user interface, **LangChain** for the RAG logic, and **Groq** for incredibly fast AI responses.

The chatbot has a unique orange and brown theme and is initially configured to answer questions about the **LangChain documentation**.


### ⚙️ Prerequisites

Before you start, make sure you have the following ready:
1.  **Python 3.9+** installed on your system.
2.  A **Groq API Key**. You can get one for free from the [Groq website](https://console.groq.com/keys).
3.  **Ollama** installed and running. Download it from the [Ollama website](https://ollama.ai/).
4.  The **`nomic-embed-text` model** pulled in Ollama. Open your terminal and run: `ollama pull nomic-embed-text`


### 🚀 Setup and Installation
Follow these steps to get the chatbot running on your computer.

#### 1\. Download the Code
First, get the project files. You can either clone the repository or download the ZIP file.

#### 2\. Create a Virtual Environment
It's a good practice to create a virtual environment to keep your project's dependencies separate.

```bash
python -m venv venv
```

#### 3\. Activate the Environment
You need to activate the environment before installing packages.

  * **On Windows (Command Prompt):**
    ```bash![WhatsApp Image 2025-09-16 at 15 44 19_2cfe77ef](https://github.com/user-attachments/assets/7eed2d65-fafc-4a00-9242-bd9095aedaf9)

    venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

#### 4\. Install Dependencies
Install all the necessary libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 🔑 Environment Variable Setup
To use the Groq API, you must save your API key in a special file.
1.  Create a new file in the same folder as `app.py` and name it **`.env`**.
2.  Open the `.env` file and add the following line, replacing `your_api_key_here` with your actual Groq API key.
    ```
    GROQ_API_KEY=your_api_key_here
    ```

### ▶️ Running the App
Make sure your Ollama application is running in the background. Then, from your terminal, run the following command to start the Streamlit app:

```bash
streamlit run app.py
```
Your browser will automatically open a new tab with the chatbot running.


### 🤖 How to Use

1.  Wait for the "Knowledge base loaded\! Ready to chat." message to appear.
2.  Type your question into the chat box at the bottom of the screen.
3.  The chatbot will retrieve relevant information from the LangChain documentation and provide an answer.


### troubleshooting

  - **"Knowledge base loading..." never finishes:** Ensure **Ollama is running** and that you have pulled the **`nomic-embed-text`** model.
  - **"Error: invalid API key" or similar:** Double-check that you have correctly added your `GROQ_API_KEY` to the `.env` file and that there are no extra spaces or characters.


### 🛠️ Reusing the Code for Other Documents

This chatbot's core logic is designed to be reusable. You can easily adapt it to work with different documents by changing just one line of code in `app.py`.

Simply replace the existing `WebBaseLoader` line with a different document loader that suits your data.

#### Examples:

  * **For a different website:**
    ```python
    from langchain_community.document_loaders import WebBaseLoader
    st.session_state.loader = WebBaseLoader("https://www.your-new-website.com")
    ```

  * **For a local PDF file:**
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    st.session_state.loader = PyPDFLoader("path/to/your-file.pdf")
    ```

  * **For an entire folder of documents:**
    ```python
    from langchain_community.document_loaders import DirectoryLoader
    st.session_state.loader = DirectoryLoader('./data', glob="*.pdf")![PHOTO3](https://github.com/user-attachments/assets/a169e634-348d-46ff-a0e7-9c99c0ce9af1)
![PHOTO2](https://github.com/user-attachments/assets/2daa280a-c4ef-4561-9b98-4acf0d553490)
![PHOTO1](https://github.com/user-attachments/assets/dc6ebf0c-04fe-4de2-bf86-57647e5e4734)
![PHOTO3](https://github.com/user-attachments/assets/9fb87826-759b-42de-b82d-3ac1de9903f6)
