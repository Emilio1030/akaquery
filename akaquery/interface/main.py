## sqlite3 related (for Streamlit)

#import pysqlite3
import sys



#import pysqlite3
__import__('pysqlite3')
import sys
import sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import json
import streamlit as st
import ipdb

# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import numpy as np
from PIL import Image
from htbuilder.units import percent, px
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts


# Start Streamlit session
st.set_page_config(page_title="Actuarial Doc Q&A Model", page_icon="📖")

st.header(
    "Actuarial Documents Q&A (RAG)"
)
st.write(
    "Please see the sidebar to select a collection of documents."
)

# Set variables
base_path = "data/pdf"


# LLM flag for augmented generation (the flag only applied to llm, not embedding model)
USE_Anthropic = True

if USE_Anthropic:
    model_name = "claude-3-sonnet-20240229"
else:
    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4-0125-preview"  # gpt-4 seems to be slow


# Define a function to scan a directory and return a dictionary of folders and files.
@st.cache_data  # Add the caching decorator
def scan_directory(base_path):
    folders_files = {}
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            files = ["All"]
            for file in os.listdir(folder_path):
                # Exclude system files like .DS_Store
                if file != ".DS_Store":
                    files.append(file)
            files[1:] = sorted(files[1:])
            folders_files[folder] = files
    return folders_files

document_list = scan_directory(base_path)
collection_list=[
    "legislation",
    "ifrs17",
    "solvencia_2",
    "reg_delegado",
    "reg_execucao"
]

@st.cache_data  # Add the caching decorator
def get_json(file_path):
    # Open and load the json file
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

#ipdb.set_trace()
summary_data = get_json("notebook/summary.json")

# ## Sidebar
# # Define the CSS for the sidebar
# sidebar_style = """
#     <style>
#     /* Change the sidebar background color */
#     [data-testid="stSidebar"] {
#         background-color: #f0f2f6;
#     }
#     </style>
#     """

# Inject the CSS into the Streamlit app
# st.markdown(sidebar_style, unsafe_allow_html=True)
st.sidebar.header("About")

st.sidebar.header("About")

with st.sidebar:
    st.markdown(
        "Welcome to **AKAQUERY**, an AI-powered tool designed to help teams and to make life easier regarding legislation consultation."
    )
    st.markdown(
        "Actuaries are strongly advised to **evaluate for accuracy** when using AI. Download the documents to read and review the source. Read the retrieved contexts to compare to AI's responses."
    )
    st.markdown("Created by [Emilio Aguiar](https://www.linkedin.com/in/matthewrwadams/).")

    # Add "Star on GitHub" link to the sidebar
    badge_html = """
    <a href="https://emilio1030.github.io/ParticleGround-Portfolio/">
    <img alt="Static Badge" src="https://img.shields.io/badge/Portfolio-Python-brightgreen?logo=python">
    </a>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("""---""")

# with st.sidebar:

#     st.header("**LLM Model**")
#     st.write(
#         f"The *{model_name}*-powered RAG process searches for and retrieves information on actuarial documents. Harness its power but **with accountability and responsibility**."
#     )
#     st.write(
#         "**AI's responses should not be relied upon as accurate or error-free.** The quality of the retrieved contexts and responses may depend on LLM algorithms, RAG parameters, and how questions are asked."
#     )
#     st.write(
#         "Actuaries are strongly advised to **evaluate for accuracy** when using AI. Download the documents to read and review the source. Read the retrieved contexts to compare to chat's response."
#     )

    collection_name = st.selectbox(
        "Select your document collection",
        collection_list,
    )

    document_name = st.selectbox(
        "Select your document",
        document_list[collection_name],
    )

    with st.expander("⚙️ RAG Parameters"):
        num_source = st.slider(
            "Top N sources to view:", min_value=4, max_value=20, value=5, step=1
        )
        flag_mmr = st.checkbox(
            "Diversity search",
            value=True,
            help="Diversity search, i.e., Maximal Marginal Relevance (MMR) tries to reduce redundancy of fetched documents and increase diversity. 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.",
        )
        _lambda_mult = st.slider(
            "Diversity parameter (lambda):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.25,
        )
        flag_similarity_out = st.checkbox(
            "Output similarity score",
            value=False,
            help="The retrieval process may become slower due to the cosine similarity calculations. A similarity score of 100% indicates the highest level of similarity between the query and the retrieved chunk.",
        )

# Create a vector store for the document collection
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="data/chroma_semantic",
    collection_name=collection_name,
)

#ipdb.set_trace()
# Retrieve and RAG chain
# Create a retriever using the vector database as the search source
search_kwargs = {"k": num_source}

# Only add the filter if the value is not "All"
if document_name != "All":
    search_kwargs["filter"] = {"source": document_name}

if flag_mmr:
    # Use MMR (Maximum Marginal Relevance) to find a set of documents
    # that are both similar to the input query and diverse among themselves
    # Increase the number of documents to get, and increase diversity
    # (lambda mult 0.5 being default, 0 being the most diverse, 1 being the least)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={**search_kwargs, "lambda_mult": _lambda_mult}
    )

else:
    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )  # use similarity search

#ipdb.set_trace()

# Chat model stream handler
class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# # Retrieval handler
# class PrintRetrievalHandler(BaseCallbackHandler):
#     def __init__(self, container, msgs, calculate_similarity=False):
#         self.status = container.status("**Context Retrieval**")
#         self.msgs = msgs
#         self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#         self.calculate_similarity = calculate_similarity

#     def on_retriever_start(self, serialized: dict, query: str, **kwargs):
#         self.status.update(label=f"**Context Retrieval:** {query}")
#         self.msgs.add_ai_message(f"Query: {query}")
#         if self.calculate_similarity:
#             self.query_embedding = self.embeddings.embed_query(query)

#     def on_retriever_end(self, documents, **kwargs):
#         source_msgs = ""
#         for idx, doc in enumerate(documents):
#             source = os.path.basename(doc.metadata["source"])
#             # page = doc.metadata["page"] + 1 # use when page-info is available
#             page_txt = ""  # if available page_txt = f", page {page}"
#             contents = doc.page_content
#             similarity_txt = ""
#             if self.calculate_similarity:
#                 content_embedding = self.embeddings.embed_query(contents)
#                 similarity = round(
#                     self.cosine_similarity(self.query_embedding, content_embedding)
#                     * 100
#                 )
#                 similarity_txt = f" \n* **Similarity score: {similarity}%**"

#             source_msg = f"# Retrieval {idx+1}\n* **Document: {source}{page_txt}**{similarity_txt}\n\n {contents}\n\n"

#             self.status.write(source_msg, unsafe_allow_html=True)
#             source_msgs += source_msg
#         self.msgs.add_ai_message(source_msgs)
#         self.status.update(state="complete")

    # def cosine_similarity(self, embedding1, embedding2):
    #     return np.dot(embedding1, embedding2) / (
    #         np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    #     )
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container, msgs, calculate_similarity=False):
        self.container = container
        self.msgs = msgs
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.calculate_similarity = calculate_similarity

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status = st.spinner(f"**Context Retrieval:** {query}")
        self.msgs.add_ai_message(f"Query: {query}")
        if self.calculate_similarity:
            self.query_embedding = self.embeddings.embed_query(query)

    def on_retriever_end(self, documents, **kwargs):
        source_msgs = ""
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            contents = doc.page_content
            similarity_txt = ""
            if self.calculate_similarity:
                content_embedding = self.embeddings.embed_query(contents)
                similarity = round(self.cosine_similarity(self.query_embedding, content_embedding) * 100)
                similarity_txt = f" \n* **Similarity score: {similarity}%**"

            with st.expander(f"📖 **Context Retrieval {idx+1}: {source}**", expanded=False):
                st.write(contents, unsafe_allow_html=True)
                if similarity_txt:
                    st.write(similarity_txt)

            source_msg = f"# Retrieval {idx+1}\n* **Document: {source}**{similarity_txt}\n\n {contents}\n\n"
            source_msgs += source_msg
        self.msgs.add_ai_message(source_msgs)

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
#ipdb.set_trace()
# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=msgs,
    return_messages=True,
)

# Setup LLM and QA chain
if USE_Anthropic:
    llm = ChatAnthropic(
        model_name=model_name,
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"],
        temperature=0,
        streaming=True,
    )
else:
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0,
        streaming=True,
    )


qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# Show the chat history
tmp_query = ""
avatars = {"human": "user" , "ai": "assistant"}

# Initialize the chat history
if len(msgs.messages) == 0:
    msgs.add_ai_message("Welcome to life actuarial document Q&A assistant!")

for msg in msgs.messages:
    if msg.content.startswith("Query:"):
        tmp_query = msg.content.lstrip("Query: ")
    elif msg.content.startswith("# Retrieval"):
        with st.expander(f"📖 **Context Retrieval:** {tmp_query}", expanded=False):
            st.write(msg.content, unsafe_allow_html=True)
    else:
        tmp_query = ""
        st.chat_message(avatars[msg.type]).write(msg.content)

# for msg in msgs.messages:
#     if msg.content.startswith("Query:"):
#         tmp_query = msg.content.lstrip("Query: ")
#     elif msg.content.startswith("# Retrieval"):
#         with st.expander(f"📖 **Context Retrieval:** {tmp_query}", expanded=False):
#             st.write(msg.content, unsafe_allow_html=True)

#     else:
#         tmp_query = ""
#         st.chat_message(avatars[msg.type]).write(msg.content)
# Download or get the main themes of the selected document

# Download or get the main themes of the selected document
if document_name != "All":
    pdf_file_path = base_path + "/" + collection_name + "/" + document_name
    document_summary = summary_data.get(collection_name, {}).get(document_name, "")
    document_summary = (
        document_summary if document_summary else "This document has no summary."
    )
    with st.sidebar:
        st.download_button(
            label="📄 Download Selected Document",
            data=open(pdf_file_path, "rb").read(),
            file_name=document_name,
            mime="application/pdf",
        )
        st.write("## Document Summary")
        st.info(document_summary)


# Download or get the main themes of the selected document
# if document_name != "All":
#     pdf_file_path = base_path + "/" + collection_name + "/" + document_name
#     document_summary = summary_data.get(collection_name, {}).get(document_name, "")
#     document_summary = (
#         document_summary if document_summary else "This document has no summary."
#     )
#     with st.sidebar:
#         st.download_button(
#             label="📄 Download Selected Document",
#             data=open(pdf_file_path, "rb").read(),
#             file_name=document_name,
#             mime="application/pdf",
#         )
#         st.write("## Document Summary")
#         st.info(document_summary)

    # if st.sidebar.button(
    #     "Get main themes of selected document",
    #     use_container_width=True,
    # ):
    #     user_query = (
    #         "What are the main themes in the document named " + document_name + "?"
    #     )
    #     st.chat_message("user").write(user_query)
    #     with st.chat_message("assistant"):
    #         retrieval_handler = PrintRetrievalHandler(
    #             st.container(), msgs, calculate_similarity=flag_similarity_out
    #         )
    #         stream_handler = StreamHandler(st.empty())
    #         response = qa_chain.run(
    #             user_query, callbacks=[retrieval_handler, stream_handler]
    #         )

# Ask the user for a question
if user_query := st.chat_input(
    placeholder="What is your question on the selected collection/document?"
):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(
            st.container(), msgs, calculate_similarity=flag_similarity_out
        )
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )

# Clear chat history or download the chat history in CSV
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:

        def clear_chat_history():
            msgs.clear()
            msgs.add_ai_message("Welcome to life actuarial document Q&A machine!")

        st.button(
            label="Clear history",
            use_container_width=True,
            on_click=clear_chat_history,
            help="Retrievals use your conversation history, which will influence future outcomes. Clear history to start fresh on a new topic.",
        )
    with col2:

        def convert_df(msgs):
            df = []
            for msg in msgs.messages:
                df.append({"type": msg.type, "content": msg.content})

            df = pd.DataFrame(df)
            return df.to_csv().encode("utf-8")

        st.download_button(
            label="Download history",
            help="Download chat history in CSV",
            data=convert_df(msgs),
            file_name="chat_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # link = "https://github.com/DanTCIM/ValAct_RAG"
    # st.caption(
    #     f"🖋️ The Python code and documentation of the project are in [GitHub]({link})."
    # )
# layout footnote
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      .stApp { bottom: 70px; }
      a { text-decoration: none; } /* Add this line to remove underline from links */
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,
        font_family="'Mulish', sans-serif",  # Add this line
    )

    style_hr = styles(
        display="block",
        margin=px(10, 10, "auto", "auto"),
        # border_style="inset",
        # border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

# function footnote
def footer():
    # LINK
    github_link = 'https://github.com/Emilio1030'
    linkedin_link = 'https://www.linkedin.com/in/emilioaguiar/'
    email_link = 'mailto:junioraguiar_83@hotmail.com'

    # ICON
    github_icon_svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" style="width: 1.28em; height: 2em; color: white;">
    <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>
    '''

    linkedin_icon_svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" style="width: 1.50em; height: 1.4em; color: white;">
    <path d="M416 32H31.9C14.3 32 0 46.5 0 64.3v383.4C0 465.5 14.3 480 31.9 480H416c17.6 0 32-14.5 32-32.3V64.3c0-17.8-14.4-32.3-32-32.3zM135.4 416H69V202.2h66.5V416zm-33.2-243c-21.3 0-38.5-17.3-38.5-38.5S80.9 96 102.2 96c21.2 0 38.5 17.3 38.5 38.5 0 21.3-17.2 38.5-38.5 38.5zm282.1 243h-66.4V312c0-24.8-.5-56.7-34.5-56.7-34.6 0-39.9 27-39.9 54.9V416h-66.4V202.2h63.7v29.2h.9c8.9-16.8 30.6-34.5 62.9-34.5 67.2 0 79.7 44.3 79.7 101.9V416z"/></svg>
    '''

    email_icon_svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" style="width: 1.55em; height: 2em; color: white;">
    <path d="M48 64C21.5 64 0 85.5 0 112c0 15.1 7.1 29.3 19.2 38.4L236.8 313.6c11.4 8.5 27 8.5 38.4 0L492.8 150.4c12.1-9.1 19.2-23.3 19.2-38.4c0-26.5-21.5-48-48-48H48zM0 176V384c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V176L294.4 339.2c-22.8 17.1-54 17.1-76.8 0L0 176z"/></svg>
    '''


    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(20), height=px(20)),
        " by ", #"with ❤️ by ",
        link("mailto:junioraguiar_83@hotmail.com", "@EmilioAguiar"),
        br(),
        "Further details on",
        a(href=github_link, target="_blank", style=styles(color="black"))(github_icon_svg),
        a(href=linkedin_link, target="_blank", style=styles(color="black"))(linkedin_icon_svg),
        a(href=email_link, target="_blank", style=styles(color="black"))(email_icon_svg),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
