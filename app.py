import streamlit as st
import os
import asyncio
import tempfile
import pandas as pd
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="chatgpt banker 🐫", layout="wide")
st.title("💬 GPT Banker com Llama, o seu analista de dados")

uploaded_file = st.file_uploader("📄 Faça upload do seu PDF ou Excel", type=["pdf", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Detecta o tipo de arquivo e carrega documentos
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        content = df.to_string(index=False)
        documents = [Document(page_content=content)]

    else:
        st.error("Formato de arquivo não suportado.")
        st.stop()

    # Divisão de texto em partes diferentes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embeddings e Vetores
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # LLM Groq com LLaMA
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    retriever = vectorstore.as_retriever()

    # Criando o PromptTemplate com restrições
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Você é um assistente de IA especialista em dados, matemática e análise documental. "
            "Sempre responda com precisão e baseando-se apenas no contexto fornecido. "
            "Explique cálculos detalhadamente e analise dados quando necessário.\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta:\n{question}"
        )
    )

    # Configura o RetrievalQA com o PromptTemplate
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Entrada de pergunta com botão
    user_question = st.text_area("❓ Faça uma pergunta sobre o arquivo", height=150)
    if st.button("📤 Enviar pergunta"):
        if user_question.strip() != "":
            with st.spinner("🧠 Pensando..."):
                response = qa_chain.run(user_question)
                st.write("🐫 Llama:", response)
        else:
            st.warning("Digite uma pergunta antes de enviar!")

    with st.expander("📚 Visualizar documentos divididos"):
        for i, doc in enumerate(texts[:5]):
            st.markdown(f"**Parte {i + 1}:**\n```\n{doc.page_content[:1000]}\n```")
