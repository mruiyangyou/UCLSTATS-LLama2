from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings



# def useRetrival(path):
#     loader =PyPDFLoader(path)
#     docs = loader.load()
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    
#     embedding = OpenAIEmbeddings()
#     db = Chroma.from_documents(docs, embedding)
#     retriver = db.as_retriever()
    
#     chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=retriver
#                                        )

#     d = chain.run('Summarise methodology of the target journal article. Please write the summary around 200 words.')
    
#     file_name =  path.split('.pdf')[0]
#     with open( file_name  + 'summary.txt', 'w') as f:
#         f.write(d)
        
# if __name__ == '__main__':
#     useRetrival('third.pdf')