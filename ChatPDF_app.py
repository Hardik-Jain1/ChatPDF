import os
import streamlit as st
import tempfile

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings


from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreToolkit
from langchain.agents.agent_toolkits import VectorStoreInfo
from PIL import Image

st.title('🔗 ChatPDF: Chat with your PDF in a conversational manner')
st.subheader('Load PDF -> Ask Questions -> Receive Answers.')

image = Image.open('chatpdf_img.jpg')
st.image(image)

st.subheader('Upload your pdf here')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

temp_file_path = os.getcwd()
        
if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)


llm = OpenAI(openai_api_key=os.environ.get("OPEN_API_KEY"), temperature=0.3, verbose=True)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN_API_KEY"))

pdf_loader = PyPDFLoader(temp_file_path)
pages = pdf_loader.load_and_split()

vstore = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

vectorstore_info = VectorStoreInfo(
    name="PDF",
    description=" A pdf file to answer your questions",
    vectorstore=vstore
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

prompt = st.text_input('Enter the input prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search - '):
        similarity_search = vstore.similarity_search_with_score(prompt) 
        st.write(similarity_search[0][0].page_content) 
        


