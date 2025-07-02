import os
import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")

groq_api_key = os.environ["GROQ_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

RAW_FAQS = [
    {"question": "What are your company's working hours?",
     "answer": "Our company operates from 9 AM to 5 PM, Monday to Friday."},
    {"question": "How can I contact customer support?",
     "answer": "Email support@example.com or call 1-800-123-4567."},
    {"question": "What products do you offer?",
     "answer": "We offer cloud platforms, data analytics, and cybersecurity solutions."},
    {"question": "Where is your main office located?",
     "answer": "123 Tech Avenue, Innovation City, CA 90210."},
    {"question": "What is your return policy?",
     "answer": "Full refunds within 30 days if product is in original condition."},
    {"question": "Do you offer international shipping?",
     "answer": "Yes, shipping cost and delivery times vary by location."},
    {"question": "What payment methods do you accept?",
     "answer": "Visa, MasterCard, AmEx, PayPal, and bank transfers."},
    {"question": "What is your company's mission statement?",
     "answer": "Empowering businesses with innovative technology solutions."},
    {"question": "How can I apply for a job?",
     "answer": "Visit our careers portal on our website."},
    {"question": "Do you provide training for your software?",
     "answer": "Yes, including online tutorials, webinars, and workshops."},
]

@st.cache_resource
def initialize_rag_with_memory():
    documents = []
    for i, faq in enumerate(RAW_FAQS):
        content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        documents.append(Document(page_content=content, metadata={"source": f"faq_{i}"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-english-v3.0"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key,
        max_tokens=150
    )

    prompt_template = PromptTemplate.from_template("""
        You are a helpful assistant. Use the following company FAQs to answer the user's question in a detailed, friendly, and conversational manner. Maintain awareness of the previous chat if relevant.

        Context:
        {context}

        Current Question: {question}
        Answer:""")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' 
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return rag_chain, memory

# rag chain
rag_chain, memory = initialize_rag_with_memory()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def handle_query(user_input):
    if not user_input.strip():
        return "Please enter a valid question."

    try:
        result = rag_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        sources = result.get("source_documents", [])
        context = "\n\n".join([f"📄 **Doc {i+1}:** {doc.page_content}" for i, doc in enumerate(sources)])
        
        full_response = f"#### Answer:\n{answer}\n\n\n---\nRetrieved Context:\n{context}"
        return full_response
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("FAQ Chatbot")

# previous conversations
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("e.g., Tell me about the company"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # generate response
    response = handle_query(prompt)
    
    # show response
    with st.chat_message("assistant"):
        st.markdown(response)
    # save history
    st.session_state.messages.append({"role": "assistant", "content": response})
