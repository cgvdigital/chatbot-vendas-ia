# --- Correção para o ChromaDB no Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- Fim da Correção ---

import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
import gspread

# --- PARTE 1: CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Assistente de Vendas IA",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Assistente de Vendas com IA")
st.caption("Conectado a uma base de dados segura no Google Sheets.")

# --- PARTE 2: CONFIGURAÇÃO DAS CHAVES E APIs ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Pega as credenciais da conta de serviço do Google Cloud
    gcp_service_account = st.secrets["gcp_service_account"]
except (KeyError, FileNotFoundError):
    st.error("ERRO: As chaves de API (GOOGLE_API_KEY ou gcp_service_account) não foram encontradas nos 'Secrets' do Streamlit. Configure-as no painel de deploy.")
    st.stop()

# --- PARTE 3: CARREGAMENTO SEGURO DOS DADOS DO GOOGLE SHEETS ---
@st.cache_data(ttl=600) # O cache expira a cada 10 minutos
def carregar_dados_da_planilha():
    try:
        # Autentica no Google Sheets usando a conta de serviço
        gc = gspread.service_account_from_dict(gcp_service_account)
        
        # ATENÇÃO: Nome da sua planilha, agora atualizado.
        NOME_DA_PLANILHA = "bd_ccdcaf" 
        planilha = gc.open(NOME_DA_PLANILHA).sheet1
        
        # Converte os dados para um DataFrame do Pandas
        df = pd.DataFrame(planilha.get_all_records())
        
        if df.empty:
            st.error("A planilha foi encontrada, mas está vazia.")
            return None
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"ERRO: Planilha com o nome '{NOME_DA_PLANILHA}' não foi encontrada. Verifique o nome e se você compartilhou a planilha com o e-mail da conta de serviço.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar com o Google Sheets: {e}")
        return None

df_dados = carregar_dados_da_planilha()

# --- PARTE 4: PROCESSAMENTO E CRIAÇÃO DO BANCO VETORIAL ---
@st.cache_resource
def criar_banco_vetorial(_df):
    if _df is None:
        return None
        
    documentos = []
    for index, row in _df.iterrows():
        texto_linha = f"Referência {index + 1}: "
        for coluna, valor in row.items():
            if valor: # Garante que só adiciona colunas com valor
                texto_linha += f"{coluna} é {valor}; "
        documentos.append(texto_linha)

    client = chromadb.Client()
    COLLECTION_NAME = "produtos_collection_segura"
    
    if len(client.list_collections()) > 0 and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    model_embedding = 'models/text-embedding-004'
    response = genai.embed_content(model=model_embedding, content=documentos, task_type="RETRIEVAL_DOCUMENT")
    
    collection.add(
        ids=[str(i) for i in range(len(documentos))],
        embeddings=response['embedding'],
        documents=documentos
    )
    return collection

if df_dados is not None:
    with st.spinner("Analisando a base de dados segura... Por favor, aguarde."):
        collection = criar_banco_vetorial(df_dados)
else:
    st.error("Não foi possível carregar o banco de dados vetorial pois os dados da planilha não foram carregados.")
    collection = None

# --- PARTE 5: LÓGICA DO CHATBOT ---
if collection is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Minha base de conhecimento foi atualizada. Como posso ajudar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Qual sua dúvida?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                model_generative = genai.GenerativeModel('gemini-1.5-flash')
                
                embedding_prompt = genai.embed_content(model='models/text-embedding-004', content=prompt, task_type="RETRIEVAL_QUERY")['embedding']
                resultados = collection.query(query_embeddings=[embedding_prompt], n_results=3)
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