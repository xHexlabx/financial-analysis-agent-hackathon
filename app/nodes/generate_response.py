from pathlib import Path
from typing import Dict, List, Union

# Import necessary components from transformers for local model loading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Assuming 'State' is defined in models.states.state
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

    # 1. Load the local model and tokenizer
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
            device_map="auto", # This tells accelerate to handle device placement
            trust_remote_code=True
        )
        model.eval()

        # --- CONFIRM GPU USAGE ---
        if torch.cuda.is_available():
            print(f"CUDA is available. Model loaded onto device: {model.device}")
            # You can also inspect specific layers:
            # for name, param in model.named_parameters():
            #     if param.is_cuda:
            #         print(f"Parameter '{name}' is on GPU: {param.device}")
            #         break # Just show one to confirm
        else:
            print("CUDA is NOT available. Model loaded onto CPU.")
        # --- END CONFIRM GPU USAGE ---

    except Exception as e:
        raise RuntimeError(f"Error loading local HuggingFace model from {LOCAL_MODEL_PATH}: {e}")

    # 2. Create a transformers pipeline
    # The 'device' argument is intentionally omitted here because 'device_map="auto"'
    # already handled device placement during model loading.
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        repetition_penalty=1.03,
        top_k=20,
        top_p=0.95,
        temperature=0.6,
        do_sample=True,
        batch_size=4, # Adjust based on your hardware
    )

    # 3. Wrap the pipeline with HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    chat = ChatHuggingFace(llm=llm)

    # New Logic for Per-Query Prompt Types
    batch_prompts = []
    for query, query_type in zip(queries, queries_type):
        prompt_template = create_answer_prompt(query_type=query_type)
        full_prompt_messages = prompt_template.invoke({"query": query})
        batch_prompts.append(full_prompt_messages)

    # Perform batch inference
    responses = chat.batch(batch_prompts)

    processed_outputs: List[str] = []
    final_answer_tag = "</think>"

    for response in responses:
        raw_content = response.content
        final_answer_index = raw_content.find(final_answer_tag)

        if final_answer_index != -1:
            processed_content = raw_content[final_answer_index + len(final_answer_tag):].strip()
        else:
            processed_content = raw_content.strip()
        processed_outputs.append(processed_content)

    return {"outputs": processed_outputs}
