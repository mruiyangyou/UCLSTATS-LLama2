from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory, ConversationSummaryMemory
import chainlit as cl
import os 
import torch
from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline

# LOAD THE VARAIABLE
_  = load_dotenv(find_dotenv())

# SET UP THE PROMPT

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def load_model():
    device_map = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained("mRuiyang/UCLStats-llama2")

    model = AutoModelForCausalLM.from_pretrained("mRuiyang/UCLStats-llama2",
                                                low_cpu_mem_usage=True,
                                                return_dict=True,
                                                torch_dtype=torch.float16,
                                                device_map=device_map,
                                                )
        
    
    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
    llm = HuggingFacePipeline(
        pipeline = pipe, 
        pipeline_kwargs={'temperature': 0}
        
    )
    
    return llm 


def create_prompt_template(include_memory = True):

    system_template =  B_SYS + """
    You are a helpful mathematics and statistics expert from UCL Statstics Department\
    that can answer questions bsaed your knowledge.

    If you feel like you don't have enough information to answer the question, say "I don't know".

    Your answers should be verbose and detailed.
    """ + E_SYS

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = B_INST + "User: Answer the following question: {question}" + E_INST
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    if include_memory:

      history = MessagesPlaceholder(variable_name='chat_history')
      prompt = ChatPromptTemplate.from_messages([system_prompt,
                                                history,
                                                human_message_prompt])
    else:
      prompt = ChatPromptTemplate.from_messages([system_prompt,
                                                human_message_prompt])

    return prompt

@cl.on_chat_start
def main():
    llm = load_model()
    
    prompt = create_prompt_template()
    
    # memory =  ConversationSummaryMemory(llm=llm, 
    #                                     return_messages=True,
    #                                     memory_key='chat_history')
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
        )
    
    cl.user_session.set('llm_chain', chain)
    
@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()





