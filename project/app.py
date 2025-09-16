import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

# --- 1. Configuration & Initial Setup ---

# Set basic Streamlit page configuration.
# THIS MUST BE THE VERY FIRST STREAMLIT COMMAND CALLED in the script.
# Removed 'icon' parameter as it might not be supported by your Streamlit version.
st.set_page_config(layout="centered", page_title="Orange & Brown RAG Chatbot")

# Load environment variables from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- 2. Custom CSS for Orange and Brown Theme ---
# This block injects custom CSS to style the Streamlit app.
st.markdown("""
<style>
    /* Import Google Font for a modern look */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    /* General App Styling */
    .stApp {
        background-color: #3D2B1F; /* Dark Brown Background */
        color: #F8F4E3; /* Light Cream Text */
        font-family: 'Roboto', sans-serif;
    }

    /* Header/Title Styling */
    h1 {
        color: #FF7F50; /* Coral Orange for Title */
        text-align: center;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); /* Subtle shadow for depth */
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #5C4033; /* Medium Brown for Sidebar */
        color: #F8F4E3;
        border-right: 2px solid #FF7F50; /* Orange border on the right */
    }
    .stSidebar h2 {
        color: #FF7F50; /* Orange for sidebar headers */
    }

    /* Chat message containers (general styling that applies to both user and assistant) */
    div.st-emotion-cache-1r6l7jp.eczjsme4 { /* Adjust selector based on Streamlit version if needed */
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); /* Stronger shadow for message bubbles */
        border: 1px solid rgba(255, 127, 80, 0.3); /* Subtle orange border */
    }

    /* User chat messages (specific styling) */
    div.st-emotion-cache-1629p8f { /* Selector for user message background */
        background-color: #7B3F00; /* Darker Orange/Brown for user */
        color: #F8F4E3;
        border-top-left-radius: 18px; /* More rounded top-left corner */
        border-top-right-radius: 18px;
        border-bottom-left-radius: 4px; /* Less rounded bottom-left corner */
        border-bottom-right-radius: 18px;
    }

    /* Assistant chat messages (specific styling) */
    div.st-emotion-cache-18ni713 { /* Selector for assistant message background */
        background-color: #8B4513; /* Saddle Brown for assistant */
        color: #F8F4E3;
        border-top-left-radius: 18px;
        border-top-right-radius: 18px;
        border-bottom-left-radius: 18px;
        border-bottom-right-radius: 4px; /* Less rounded bottom-right corner */
    }

    /* Chat input text area and container */
    div.st-emotion-cache-1r4qj8m { /* Selector for the chat input container */
        background-color: #2F1E14; /* Very dark brown for input */
        border-radius: 15px;
        border: 1px solid #FF7F50; /* Orange border for input */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    }
    .st-emotion-cache-1ae43m8 textarea { /* Target the actual textarea for text color */
        color: #F8F4E3;
        background-color: transparent; /* Make textarea background transparent */
    }

    /* Customizing Streamlit buttons */
    .stButton>button {
        background-color: #FF7F50; /* Orange button background */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease; /* Smooth hover effect */
    }
    .stButton>button:hover {
        background-color: #E86A3C; /* Slightly darker orange on hover */
        color: #F8F4E3;
    }

    /* Styling for Streamlit info/warning boxes */
    .stAlert {
        background-color: #FFDAB9; /* Peach Puff for alerts */
        color: #3D2B1F; /* Dark text on light background */
        border-radius: 10px;
        border: 1px solid #E86A3C;
    }

    /* Hide the default Streamlit header/footer for a cleaner look */
    header.st-emotion-cache-1dp5c9u { visibility: hidden; height: 0px; }
    .st-emotion-cache-ch5fge { visibility: hidden; height: 0px; }
    .st-emotion-cache-h4xjwx { visibility: hidden; } /* Hide hamburger menu */

</style>
""", unsafe_allow_html=True)


st.title("🍂 CogniChat: Your RAG Assistant") # Moved title here after set_page_config

# --- 3. RAG Component Initialization (Using Streamlit Session State for Efficiency) ---
# This block runs only once when the app starts or "vectors" is not yet in session_state.
with st.spinner("Loading knowledge base..."): # Show a spinner during initial loading
    if "vectors" not in st.session_state:
        # Initialize Ollama Embeddings for vector creation
        st.session_state.embeddings = OllamaEmbeddings()
        # Load documents from a web URL (replace with PDF loader if needed for local files)
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()
        
        # Split documents into smaller chunks for better retrieval
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # Process only the first 40 documents for demonstration (adjust as needed)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:40])
        # Create a FAISS vector store from the documents and embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("Knowledge base loaded! Ready to chat.") # Success message after loading

# --- 4. LLM and Chain Setup ---
# Initialize ChatGroq model with your API key and a supported model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192") # Corrected model name

# Define the chat prompt template for RAG
prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only. Please provide the most accurate response.
    <context>
    {context}
    </context>
    Question:{input}
    """
)

# Create a chain to combine documents based on the prompt
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create a retriever from the FAISS vector store
retriever = st.session_state.vectors.as_retriever()

# Create the final retrieval chain that combines retrieval and document stuffing
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 5. Chat History Management ---
# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on each app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # Use Streamlit's built-in chat message container
        st.markdown(message["content"])

# --- 6. User Input and Response Generation ---
# st.chat_input creates an input box at the bottom for chat-like interaction
user_input = st.chat_input("Ask me a question about LangChain documentation...")

if user_input:
    # Add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer..."): # Show a thinking indicator
            start_time = time.process_time() # Record start time for response time calculation
            response = retrieval_chain.invoke({'input': user_input})
            end_time = time.process_time() # Record end time
            ai_response = response['answer']
            st.markdown(ai_response) # Display AI response
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            # Display response time subtly
            st.caption(f"Response time: {end_time - start_time:.2f} seconds")

            # Optional: Display source documents in an expander for transparency
            with st.expander("🔍 See Source Documents"): # Added a search icon for visual cue
                st.info("Information used from the following document chunks:") # Info box for explanation
                for i, doc in enumerate(response["context"]): # Loop through retrieved document chunks
                    st.write(f"**Chunk {i+1}:**") # Display chunk number
                    st.write(doc.page_content) # Display the content of the chunk
                    if doc.metadata and 'source' in doc.metadata: # Check if source metadata exists
                        st.caption(f"Source: {doc.metadata['source']}") # Display the source URL
                    st.write("---") # Separator between chunks

# --- 7. Additional UI Elements (Sidebar) ---
# The 'with st.sidebar:' block creates content for the sidebar
with st.sidebar:
    st.header("App Info")
    st.write("This is a demo of a **Retrieval-Augmented Generation (RAG)** chatbot.")
    st.write("It leverages **LangChain** for orchestration and **Groq** for high-speed AI responses.")
    st.write("Information is retrieved from the **LangChain documentation website**.")

    st.warning("⚠️ **Important:** Ensure your `GROQ_API_KEY` is set in your `.env` file and **Ollama is running** locally with `nomic-embed-text` (or another embedding model) pulled for embeddings!")

    # Button to clear chat history
    if st.button("🔄 Clear Chat History"): # Added a refresh icon
        st.session_state.messages = [] # Clear messages in session state
        st.rerun() # Rerun the app to refresh the display

    st.info("💡 **Tip:** Try asking questions like 'What is LangChain?', 'Explain RAG?', or 'How to use Chains?'")
