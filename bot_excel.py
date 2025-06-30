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
from google_auth_oauthlib.flow import Flow
import os

# --- PARTE 1: LÓGICA DE AUTENTICAÇÃO ---
st.set_page_config(page_title="Chatbot Corporativo MV", page_icon="🤖", layout="wide")

def create_google_auth_flow():
    # Garante que o redirect_uri está nos secrets, senão usa localhost
    redirect_uri = st.secrets.get("REDIRECT_URI", "http://localhost:8501")
    
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
        redirect_uri=redirect_uri,
    )

def show_login_page():
    st.title("Bem-vindo ao Chatbot Corporativo MV")
    st.write("Por favor, faça o login com sua conta Google (@mv.com.br) para continuar.")
    
    flow = create_google_auth_flow()
    authorization_url, _ = flow.authorization_url()
    
    st.link_button("Login com Google", authorization_url)

def handle_auth_callback():
    code = st.query_params.get("code")
    if code and "email" not in st.session_state:
        try:
            flow = create_google_auth_flow()
            flow.fetch_token(code=code)
            credentials = flow.credentials
            user_info = credentials.id_token
            st.session_state.email = user_info.get("email")
        except Exception as e:
            st.error(f"Erro durante a autenticação: {e}")
            st.session_state.email = None # Limpa em caso de erro

# --- LÓGICA PRINCIPAL DO APP ---
if "email" not in st.session_state:
    st.session_state.email = None

handle_auth_callback()

if st.session_state.email is None:
    show_login_page()
elif not st.session_state.email.endswith("@mv.com.br"):
    st.error("Acesso Negado. Este aplicativo é restrito a usuários com e-mail do domínio @mv.com.br.")
    st.write(f"Você está logado como: {st.session_state.email}")
    if st.button("Fazer logout"):
        st.session_state.email = None
        st.query_params.clear()
        st.rerun()
else:
    # --- SE O LOGIN FOR VÁLIDO, O CHATBOT É EXIBIDO ---
    
    st.sidebar.success(f"Logado como: {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.email = None
        st.query_params.clear()
        st.rerun()

    st.title("🤖 Chatbot Corporativo MV")
    st.caption("Conectado a uma base de dados segura no Google Sheets.")

    # --- O restante do seu código do chatbot começa aqui ---
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
        gcp_service_account = st.secrets["gcp_service_account"]
    except (KeyError, FileNotFoundError):
        st.error("ERRO: As chaves de API do backend não foram encontradas.")
        st.stop()

    @st.cache_data(ttl=600)
    def carregar_dados_da_planilha():
        try:
            gc = gspread.service_account_from_dict(gcp_service_account)
            NOME_DA_PLANILHA = "bd_ccdcaf" 
            planilha = gc.open(NOME_DA_PLANILHA).sheet1
            df = pd.DataFrame(planilha.get_all_records())
            if df.empty: return None
            return df
        except Exception as e:
            st.error(f"Erro ao conectar com o Google Sheets: {e}")
            return None

    df_dados = carregar_dados_da_planilha()

    @st.cache_resource
    def criar_banco_vetorial(_df):
        if _df is None: return None
        documentos = []
        for index, row in _df.iterrows():
            texto_linha = f"Referência {index + 1}: "
            for coluna, valor in row.items():
                if valor: texto_linha += f"{coluna} é {valor}; "
            documentos.append(texto_linha)

        client = chromadb.Client()
        COLLECTION_NAME = "produtos_collection_segura_mv"
        if len(client.list_collections()) > 0 and COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(name=COLLECTION_NAME)
        collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        model_embedding = 'models/text-embedding-004'
        response = genai.embed_content(model=model_embedding, content=documentos, task_type="RETRIEVAL_DOCUMENT")
        collection.add(ids=[str(i) for i in range(len(documentos))], embeddings=response['embedding'], documents=documentos)
        return collection

    if df_dados is not None:
        collection = criar_banco_vetorial(df_dados)
    else:
        collection = None

    if collection is not None:
        if "messages" not in st.session_state or len(st.session_state.messages) == 0:
            st.session_state.messages = [{"role": "assistant", "content": "Olá! Autenticação bem-sucedida. Como posso ajudar?"}]

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
                    prompt_final = f"CONTEXTO:\n{contexto}\n\nHISTÓRICO DA CONVERSA:\n{historico_formatado}\n\nPERGUNTA ATUAL do usuário:\n{prompt}\n\nCom base em tudo isso, forneça uma resposta útil e concisa:"
                    response = model_generative.generate_content(prompt_final)
                    resposta = response.text
                    st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})