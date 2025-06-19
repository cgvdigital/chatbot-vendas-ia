# --- Correção para o ChromaDB no Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- Fim da Correção ---

# O resto do seu código começa aqui...
import pandas as pd
import google.generativeai as genai
import chromadb
import streamlit as st
import time
import os

# --- PARTE 1: CONFIGURAÇÃO E TÍTULO DA PÁGINA ---

# Configura o título da página, ícone e layout
st.set_page_config(
    page_title="Assistente de Vendas IA",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Assistente de Vendas com IA")
st.caption("Faça perguntas sobre os produtos da nossa base de dados.")

# --- PARTE 2: CARREGAMENTO DOS DADOS E SEGURANÇA DA CHAVE (IMPORTANTE!) ---

# Pede para o usuário inserir a chave de API de forma segura
try:
    # Tenta obter a chave dos segredos do Streamlit (para quando estiver online)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("ERRO: Chave de API não encontrada nos 'Secrets' do Streamlit. Por favor, configure-a no painel de deploy.")
    st.stop()

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("Chave de API configurada, mas vazia.")
    st.stop()

# --- PARTE 3: PROCESSAMENTO DA BASE DE DADOS (SÓ EXECUTA UMA VEZ) ---

@st.cache_resource
def carregar_dados():
    EXCEL_FILE_PATH = 'minha_base.xlsx'
    COLLECTION_NAME = "produtos_collection"
    
    if not os.path.exists(EXCEL_FILE_PATH):
        st.error(f"ERRO: O arquivo de dados '{EXCEL_FILE_PATH}' não foi encontrado no repositório.")
        st.stop()
        
    df = pd.read_excel(EXCEL_FILE_PATH)
    documentos = []
    for index, row in df.iterrows():
        texto_linha = f"Referência {index + 1}: "
        for coluna, valor in row.items():
            texto_linha += f"{coluna} é {valor}; "
        documentos.append(texto_linha)

    client = chromadb.Client()
    
    if len(client.list_collections()) > 0 and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    model_embedding = 'models/text-embedding-004'
    response = genai.embed_content(model=model_embedding, content=documentos)
    
    collection.add(
        ids=[str(i) for i in range(len(documentos))],
        embeddings=response['embedding'],
        documents=documentos
    )
    
    return collection

# Carrega a coleção (o "cérebro" do bot)
with st.spinner("Analisando a base de dados... Por favor, aguarde."):
    collection = carregar_dados()

# --- PARTE 4: LÓGICA DO CHATBOT COM MEMÓRIA ---

# Inicializa o histórico da conversa na sessão
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com os nossos produtos hoje?"}]

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a pergunta do usuário
if prompt := st.chat_input("Qual sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            model_generative = genai.GenerativeModel('gemini-1.5-flash')
            
            embedding_prompt = genai.embed_content(model='models/text-embedding-004', content=prompt)['embedding']
            resultados = collection.query(query_embeddings=[embedding_prompt], n_results=5)
            contexto = "\n".join(resultados['documents'][0])
            
            historico_formatado = "\n".join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
            
            prompt_final = f"""
            Você é um assistente de vendas e deve responder com base no CONTEXTO e no HISTÓRICO da conversa.

            CONTEXTO:
            {contexto}

            HISTÓRICO DA CONVERSA:
            {historico_formatado}

            PERGUNTA ATUAL do usuário:
            {prompt}
            
            Com base em tudo isso, forneça uma resposta útil e concisa:
            """
            
            response = model_generative.generate_content(prompt_final)
            resposta = response.text
            st.markdown(resposta)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta})
