from pathlib import Path
from typing import Dict, List, Union

# Import necessary components from transformers for local model loading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Assuming 'State' is defined in models.states.state
# Now, 'state' will contain 'queries': List[str] and 'queries_type': List[int]
from models.states.state import State

## Prompt Configuration
PROMPT_FILES: Dict[int, Path] = {
    0: Path("app/prompts/th_choices.txt"),
    1: Path("app/prompts/eng_choices.txt"),
    2: Path("app/prompts/th_rise_fall.txt"),
    3: Path("app/prompts/eng_rise_fall.txt"),
}

# --- Configuration for Local HuggingFace Model ---
# IMPORTANT: Replace with the actual path to your local model
# This path should point to a directory containing model, tokenizer, and config files.
LOCAL_MODEL_PATH = "Qwen/Qwen3-8B" # e.g., "./local_models/Qwen/Qwen3-8B"

def create_answer_prompt(query_type: int) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate based on the given query type.

    Args:
        query_type (int): An integer representing the type of query,
                          which maps to a specific prompt file.

    Returns:
        ChatPromptTemplate: A LangChain ChatPromptTemplate object.

    Raises:
        ValueError: If an invalid query_type is provided or the prompt file is not found.
    """
    prompt_file_path = PROMPT_FILES.get(query_type)

    if not prompt_file_path:
        raise ValueError(f"Invalid query_type: {query_type}. No prompt file defined for this type.")

    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

    try:
        system_prompt = prompt_file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError(f"Error reading prompt file {prompt_file_path}: {e}")

    # For batch inference, the human part will be iterated over multiple queries
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")])


def generate_response(state: State) -> Dict[str, List[str]]:
    """
    Generates responses from a local LLM for a batch of queries,
    where each query can have a distinct prompt type.

    Args:
        state (State): The current state containing:
                       - 'queries': A list of strings, each being a query.
                       - 'queries_type': A list of integers, where each integer
                                         corresponds to the query_type for the
                                         respective query in 'queries'.
                                         Must have the same length as 'queries'.

    Returns:
        Dict[str, List[str]]: A dictionary containing a list of processed LLM outputs.
    """
    queries: List[str] = state['queries']
    queries_type: List[int] = state['queries_type']

    if len(queries) != len(queries_type):
        raise ValueError("The length of 'queries' must match the length of 'queries_type'.")

    # 1. Load the local model and tokenizer (unchanged from previous version)
    try:
        config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        if hasattr(config, '_pre_quantization_config') and 'load_in_bf16' in config._pre_quantization_config and config._pre_quantization_config['load_in_bf16']:
             torch_dtype = torch.bfloat16
        else:
             torch_dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading local HuggingFace model from {LOCAL_MODEL_PATH}: {e}")

    # 2. Create a transformers pipeline (unchanged from previous version)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        repetition_penalty=1.03,
        top_k=20,
        top_p=0.95,
        temperature=0.6,
        do_sample=True,
        batch_size=8, # Adjust based on your hardware
        device=0 if torch.cuda.is_available() else -1
    )

    # 3. Wrap the pipeline with HuggingFacePipeline (unchanged from previous version)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    chat = ChatHuggingFace(llm=llm)

    # --- New Logic for Per-Query Prompt Types ---
    batch_prompts = []
    # Zip queries and their corresponding types to create prompts individually
    for query, query_type in zip(queries, queries_type):
        prompt_template = create_answer_prompt(query_type=query_type)
        full_prompt_messages = prompt_template.invoke({"query": query})
        batch_prompts.append(full_prompt_messages)

    # Perform batch inference
    responses = chat.batch(batch_prompts)

    processed_outputs: List[str] = []
    final_answer_tag = "Final Answer:"

    for response in responses:
        raw_content = response.content
        final_answer_index = raw_content.find(final_answer_tag)

        if final_answer_index != -1:
            processed_content = raw_content[final_answer_index + len(final_answer_tag):].strip()
        else:
            processed_content = raw_content.strip()
        processed_outputs.append(processed_content)

    return {"outputs": processed_outputs}
