import os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import csv

# set data directory
cwd = os.getcwd()
raw_data_path = os.path.join(cwd, '../data/raw')
output_data_path = os.path.join(cwd, '../data/processed/train')

# cout

def file_processing(file_name):
    file_path = os.path.join(raw_data_path, file_name)
    loader = PyPDFLoader(file_path)
    text = loader.load()
    
    question_gen = ''
    
    for d in text:
        question_gen += d.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    
    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 100
    )
    
    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)
    
    return document_ques_gen, document_ans_gen


def llm_pipeline(file_name):
    document_ques_gen, document_ans_gen = file_processing(file_name)
    
    llm_ques_gen = ChatOpenAI(
        temperature=0.3
    )
    
    prompt_template = """
    You are an expert at creating questions based on statistics materials and documentation.
    Your goal is to prepare an examination paper about statistics concepts and applications.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the studenst for exams.
    Make sure not to lose any important information.

    QUESTIONS:
    """
    
    PROMPT_QUES = PromptTemplate(template=prompt_template, input_variables=['text'])
    
    refine_template = """
    You are an expert at creating practice questions based on statistics material and documentation.
    Your goal is to help students prepare for a test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    
    
    REFINE_PROMPT_QUES = PromptTemplate(input_variables=['existing_answer', 'text'],
                                    template=refine_template)
    
    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUES, 
                                            refine_prompt=REFINE_PROMPT_QUES)
    
    ques = ques_gen_chain.run(document_ques_gen)
    
    # answer the question
    embedding = OpenAIEmbeddings()
    
    vector_store = FAISS.from_documents(document_ans_gen, embedding)
    
    llm_ans_chain = ChatOpenAI(
        temperature=0.1
    )
    
    ques_list = ques.split('\n')
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
    
    answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_ans_chain, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())
    
    return answer_gen_chain, filtered_ques_list

def main():
    
    output_file_path = os.path.join(output_data_path, 'train.csv')
    with open(output_file_path, 'w', newline="", encoding='utf-8') as csvfile:
        file_list = os.listdir(raw_data_path)
        for f in file_list:
            answer_gen_chain, filtered_ques_list = llm_pipeline(f)
            csv_writter = csv.write(csvfile)
            csv_writter.writerow(['question', 'answer'])
            for question in filtered_ques_list:
                answer = answer_gen_chain.run(question)
                csv_writter.writerow([question, answer])
                
         
if __name__ == '__main__':
    main()
    
    