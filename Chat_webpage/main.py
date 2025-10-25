import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


st.title("Ask Questions about any Webpage.....")
st.caption("This app allows you to chat with a webpage using local llama3 and RAG")

## Webpage URL input
url = st.text_input("Enter URL.......", type="default")

# Ollma 
model = "llama3"
llm = ChatOllama(model=model)

# if webpage URL is provided
if url:
    # Load the data
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    ## Splitting text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    pages = splitter.split_documents(docs)
    
    
    #  Embeddings and vs store
    embeddings = OllamaEmbeddings(model = "llama3")
    vector_store = Chroma.from_documents(pages, embedding = embeddings)
    
    ## Call Ollama 
    def ollama_chat(question, context):
        prompt = f"""You are an AI assistant. Use the following context to answer the question.
        
        Context: {context} \n\n
        
        Question: {question}
        """
        
        response = llm.invoke(prompt)
        return response.content

    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    
    ## Combine all relevant docs
    def combine_docs(docs):
        combined_text = "\n\n".join([d.page_content for d in docs])
        return combined_text
    
    
    ## Rag chain
    def rag_chain(question):
        relevant_docs = retriever.invoke(question)
        formatted_context = combine_docs(relevant_docs)
        answer = ollama_chat(question, formatted_context)
        return answer
    
    
        # Ask a question about the webpage
    prompt = st.text_input("Ask any question about the webpage")

    # Chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)