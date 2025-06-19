# Importa todas as ferramentas necessárias
import pandas as pd
import google.generativeai as genai
import chromadb
import streamlit as st # Importa o Streamlit
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
# Isso evita deixar a chave exposta no código
try:
    # Tenta obter a chave dos segredos do Streamlit (para quando estiver online)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # Se não encontrar, pede para o usuário no app
    GOOGLE_API_KEY = st.text_input("Digite sua Chave de API do Google AI Studio:", type="password")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("Por favor, insira sua chave de API para continuar.")
    st.stop() # Interrompe a execução se a chave não for fornecida

# --- PARTE 3: PROCESSAMENTO DA BASE DE DADOS (SÓ EXECUTA UMA VEZ) ---

# Usamos @st.cache_resource para garantir que esta parte rode apenas uma vez,
# mesmo que o usuário interaja com a página. Isso economiza tempo e processamento.
@st.cache_resource
def carregar_dados():
    EXCEL_FILE_PATH = 'minha_base.xlsx'
    COLLECTION_NAME = "produtos_collection"
    
    if not os.path.exists(EXCEL_FILE_PATH):
        st.error(f"ERRO: O arquivo '{EXCEL_FILE_PATH}' não foi encontrado.")
        st.stop()
        
    df = pd.read_excel(EXCEL_FILE_PATH)
    documentos = []
    for index, row in df.iterrows():
        texto_linha = f"Referência {index + 1}: "
        for coluna, valor in row.items():
            texto_linha += f"{coluna} é {valor}; "
        documentos.append(texto_linha)

    client = chromadb.Client() # Cliente em memória para simplificar
    
    # Deleta a coleção se ela já existir para garantir dados atualizados
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
    
    # Esta mensagem agora aparecerá na interface web, não no terminal
    st.success("Base de dados carregada e processada com sucesso!")
    return collection

# Carrega a coleção (o "cérebro" do bot)
collection = carregar_dados()

# --- PARTE 4: LÓGICA DO CHATBOT COM MEMÓRIA ---

# Inicializa o histórico da conversa na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a pergunta do usuário
if prompt := st.chat_input("Qual sua dúvida?"):
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Monta o prompt para o Gemini com o histórico da conversa
    model_generative = genai.GenerativeModel('gemini-1.5-flash')
    
    # Busca no ChromaDB por contexto relevante
    embedding_prompt = genai.embed_content(model='models/text-embedding-004', content=prompt)['embedding']
    resultados = collection.query(query_embeddings=[embedding_prompt], n_results=5)
    contexto = "\n".join(resultados['documents'][0])
    
    # Constrói um histórico formatado para o modelo
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
    
    # Gera e exibe a resposta do bot
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = model_generative.generate_content(prompt_final)
            resposta = response.text
            st.markdown(resposta)
    
    # Adiciona a resposta do bot ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta})