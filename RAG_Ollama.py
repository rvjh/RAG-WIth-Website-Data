from langchain_community.llms import Ollama
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  ## for splitting documents
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""


llm = Ollama(model="mistral", temperature=0)

loader = WebBaseLoader(
    web_path="http://www.columbia.edu/~fdc/sample.html")

## Loading the whole document
docs = loader.load()

## creating splits
text_splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,  # to add a metadata attribute
)

all_splits = text_splits.split_documents(docs)  ## all documents splitting

## index storing to search at runtime

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

## now retriver chain to retrive ans for a query

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}   ## 4 relevent splits
)

#retriever.get_relevant_documents("what is list in html ?")  ## to see the general output

## lets create a functions to join all relevent documents

def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)

## now generating the output -> a chain which will take a question, retrive a relevent document
## construct a prompt, pass to the model and get the output -> getting a rag with langchain hub

prompt = hub.pull("rlm/rag-prompt")

## now create a chain -> first context

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

a = rag_chain.invoke("what is List in Html ?")  ## Asking questions
print(a)
