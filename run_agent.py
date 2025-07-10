# run_agent.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
from lc_tools import FetchTechTweets, PredictSentiment

def main():
    # Expand the user path to the locally downloaded model
    model_path = os.path.expanduser(
        "*******************************************************************************"
    )

    # Load tokenizer and model using Auto classes
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    # Create a text-generation pipeline for the local model
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1
    )

    # Wrap the HuggingFace pipeline in a LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=text_gen)

    # Register your custom tools
    tools = [FetchTechTweets(), PredictSentiment()]

    # Initialize the agent with your local LLM and tools
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Run the agent: it will decide which tool to call and in what order
    result = agent.run("Fetch recent #Tech tweets and classify their sentiment.")
    print(result)

if __name__ == "__main__":
    main()
