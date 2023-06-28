import json
import argparse
import tiktoken
from context import ChatContext

from llama_cpp import Llama
import sys

enc = tiktoken.get_encoding("cl100k_base")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="airoboros-33b-gpt4-1.4.ggmlv3.q2_K.bin")
parser.add_argument("-l", "--len", type=int, default=2096)
parser.add_argument("-s", "--stream", type=bool, default=False)

args = parser.parse_args()

CONTEXT_END_BUFF = 512                                              # The difference between the max context len and when we should start pruning chat history
CONTEXT_LEN = args.len                                              # Limits of the model
CONTEXT_LEN_HIGHWATERMARK = CONTEXT_LEN - CONTEXT_END_BUFF        # Set the context length of the model

llm = Llama(model_path=args.model, n_threads=14, n_gpu_layers=42, seed=-1, n_ctx=2048)

context = ChatContext(
    initial_context="A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.",
    encoder=enc,
    max_length=CONTEXT_LEN_HIGHWATERMARK
)

while(True):
    user_input = input("User> ")
    context.add_user_input(user_input)

    llm_str = context.get_context_str()
    stream = llm(
        llm_str,
        max_tokens=1800,
        repeat_penalty= 1.1,
        stop=[" USER: "],
        stream=False,
    )

    string = stream["choices"][0]["text"]
    context.add_assistant_output(string)
    print(string)
    print()
