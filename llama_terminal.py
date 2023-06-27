import json
import argparse
import tiktoken

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

context = ["A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."]
context_ind = 0

while(True):

    user_input = " USER: " + input("User> ")
    context.append(user_input)
    context_ind += 1

    context.append(" ASSISTANT:")
    context_ind += 1


    #Check if we need to prune the chat context
    while len(enc.encode(user_input)) > CONTEXT_LEN_HIGHWATERMARK:
        context.pop(1)                                              # Pop the oldest request (not including model pre-amble)

    llm_str = "".join(context)
    stream = llm(
        llm_str,
        max_tokens=512,
        repeat_penalty= 1.1,
        stop=[" USER: "],
        stream=False,
    )

    string = stream["choices"][0]["text"]
    context[context_ind] += string
    print(string)
    print()

