import discord
from discord.ext import commands
from context import ChatContext
import tiktoken
from llama_cpp import Llama
import argparse
import os
import asyncio
import time

# Initialize the encoder
enc = tiktoken.get_encoding("cl100k_base")
INITIAL_CONTEXT = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."

model_dict = {1 : "nous-hermes-13b.ggmlv3.q2_K.bin", 2 : "airoboros-33b-gpt4-1.4.ggmlv3.q2_K.bin", 3 : "guanaco-65B.ggmlv3.q2_K.bin"}

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="guanaco-65B.ggmlv3.q2_K.bin")
parser.add_argument("-l", "--len", type=int, default=2048)
parser.add_argument("-s", "--stream", type=bool, default=False)

args = parser.parse_args()

CONTEXT_END_BUFF = 512                                              # The difference between the max context len and when we should start pruning chat history
CONTEXT_LEN = args.len                                              # Limits of the model
CONTEXT_LEN_HIGHWATERMARK = CONTEXT_LEN - CONTEXT_END_BUFF        # Set the context length of the model

llm = Llama(model_path=args.model, n_threads=14, n_gpu_layers=43, seed=-1, n_ctx=2048)

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize an empty dictionary to store contexts
contexts = {}

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
        # Update the bot's nickname in all servers
    for guild in bot.guilds:
        me = guild.me
        await me.edit(nick="Llama Bot")


@bot.event
async def on_message(message):
    if message.channel.name != 'chatgpt':
        return

    if message.author == bot.user:
        return

    if message.content[0] == '!':
       await bot.process_commands(message)

    else:
        # Get the server ID
        guild_id = message.guild.id

        # If this server doesn't have a context yet, create one
        if guild_id not in contexts:
            contexts[guild_id] = ChatContext(INITIAL_CONTEXT, enc, 2048)


        # Add the message to the context
        contexts[guild_id].add_user_input(message.content)

        # Get the context for this server
        async with message.channel.typing():
            llm_str = contexts[guild_id].get_context_str()

            stream = llm(
                llm_str,
                max_tokens=512,
                repeat_penalty= 1.1,
                stop=[" USER: "],
                stream=True,
            )
        
        bot_message = None
        string = ""
        start_time = time.time()
        for outputs in stream:
            string += outputs["choices"][0]['text']

            if bot_message is None and len(string.strip(" ")) > 0:
                bot_message = await message.channel.send(string)

            if (time.time() - start_time) >= 1 and bot_message is not None:
                start_time = time.time()
                await bot_message.edit(content=string)

        if bot_message is not None:
            await bot_message.edit(content=string)
        else:
            bot_message = await message.channel.send("*stares at you akwardly...")

        contexts[guild_id].add_assistant_output(string)



@bot.command()
async def reset_chat(ctx):
        
    if ctx.channel.name != 'chatgpt':
        return
    
    guild_id = ctx.guild.id
        # If this server doesn't have a context yet, create one
    if guild_id not in contexts:
        contexts[guild_id] = ChatContext(INITIAL_CONTEXT, enc, 2048)

    contexts[guild_id].reset_context()
    await ctx.channel.send("RESET CHAT HISTORY (THE BOT HAS FORGETTON THE CONVERSATION SO FAR)")

@bot.command()
async def set_initial_context(ctx, *, initial_context):
        
    if ctx.channel.name != 'chatgpt':
        return
    
    guild_id = ctx.guild.id
        # If this server doesn't have a context yet, create one
    if guild_id not in contexts:
        contexts[guild_id] = ChatContext(INITIAL_CONTEXT, enc, 2048)

    contexts[guild_id].set_initial_context(initial_context)
    await ctx.channel.send("THE BOT'S PERSONALITY AND THE CHAT IS NOW INFLUENCED BY TO YOUR TEXT: \n\n" + initial_context)

    
@bot.command()
async def switch_model(ctx, model):
        
    global llm                              # Ensure we are accessing the global llm variable

    if ctx.channel.name != 'chatgpt':
        return
    
    guild_id = ctx.guild.id
    # If this server doesn't have a context yet, create one
    if guild_id not in contexts:
        contexts[guild_id] = ChatContext(INITIAL_CONTEXT, enc, 2048)
    model_str = model_dict[int(model)]
    del llm
    llm = Llama(model_path=model_str, n_threads=14, n_gpu_layers=43, seed=-1, n_ctx=2048)

    await ctx.channel.send("THE BOT IS NOW RUNNING ON A DIFFERENT MODEL: " + model_str)


token = os.environ['DISCORD_TOKEN']
bot.run(token)
