from app import create_prompt_template, load_model
import os
import nest_asyncio
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chains import LLMChain

nest_asyncio.apply()
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "uclstatsllama2"

#### Better run the code line by line Using Vscode's Python Interactive Shift + Enter ###

# test the llm
client = Client()
llm = load_model()
llm.run("What is normal distributuin?")

# test the data without label
inputs = [
    'What is an estimate and what is estimator?',
    'Explain the difference between them',
    'What is law of big number?'
]

dataset_name = 'UCLLLama2-test'
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="UCLllama-2 test",
)

for input_prompt in inputs:
    # Each example must be unique and have inputs defined.
    # Outputs are optional
    client.create_example(
        inputs={"question": input_prompt},
        outputs=None,
        dataset_id=dataset.id,
    )
    
def chain_constructor():
    prompt = create_prompt_template(include_memory = False)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
        )
    return chain

    
    
eval_config = RunEvalConfig(
    evaluators=[
        # You can specify an evaluator by name/enum.
        # In this case, the default criterion is "helpfulness"
        "criteria",
        # Or you can configure the evaluator
        RunEvalConfig.Criteria("harmfulness"),
        RunEvalConfig.Criteria("misogyny"),
        RunEvalConfig.Criteria(
            {
                "cliche": "Are the answers cliche? "
                "Respond Y if they are, N if they're entirely unique."
            }
        ),
    ]
)

chain_results = client.run_on_dataset(
    dataset_name="UCLLLama2-test",
    llm_or_chain_factory=chain_constructor,
    evaluation=eval_config,
    project_name="test-no-label",
    verbose=True,
   
)


# with label
example_inputs = [
    ('What is the distribution of sample mean?', 'Normal Distribution'),
    ('What is the mean of sampling distribution of sample mean?', 'Population mean')
]


dataset_name = "Sample stats question"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Questions and answers about simple statistics question",
)

for input_prompt, output_answer in example_inputs:
    client.create_example(
        inputs={"question": input_prompt},
        outputs={"answer": output_answer},
        dataset_id=dataset.id,
    )
    
evaluation_config = RunEvalConfig(
    evaluators=[
        "qa",  # correctness: right or wrong
        "context_qa",  # refer to example outputs
        "cot_qa",  # context_qa + reasoning
    ], 
    input_key = 'question'
)

chain_results = client.run_on_dataset(
    dataset_name="Sample stats question",
    llm_or_chain_factory=chain_constructor,
    evaluation=evaluation_config,
    project_name="test-label",
    verbose=True,
)
