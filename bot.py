import discord
from discord.ext import commands
from context import ChatContext
import tiktoken
from llama_cpp import Llama
import argparse
import os
import asyncio
import time
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Initialize the encoder
enc = tiktoken.get_encoding("cl100k_base")
INITIAL_CONTEXT = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."
STREAM_UPDATE_PERIOD = 1.2          # Seconds between updates

model_dict = {1 : "nous-hermes-13b.ggmlv3.q2_K.bin", 2 : "airoboros-33b-gpt4-1.4.ggmlv3.q2_K.bin",  3 : "guanaco-65B.ggmlv3.q2_K.bin" }

prompt_format_dict = {"nous-hermes-13b.ggmlv3.q2_K.bin": ("### Instruction: ", "### Response: "), "airoboros-33b-gpt4-1.4.ggmlv3.q2_K.bin": ("USER: ", "Assistant: "), "guanaco-65B.ggmlv3.q2_K.bin": ("USER: ", "Assistant: ")}

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="airoboros-33b-gpt4-1.4.ggmlv3.q2_K.bin")
parser.add_argument("-l", "--len", type=int, default=2048)
parser.add_argument("-s", "--stream", type=bool, default=False)

args = parser.parse_args()

CONTEXT_END_BUFF = 512                                              # The difference between the max context len and when we should start pruning chat history
CONTEXT_LEN = args.len                                              # Limits of the model
CONTEXT_LEN_HIGHWATERMARK = CONTEXT_LEN - CONTEXT_END_BUFF          # Set the context length of the model

llm = Llama(model_path=args.model, n_threads=14, n_gpu_layers=38, seed=-1, n_ctx=2048)

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize an empty dictionary to store contexts
contexts = {}
model_str = args.model

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
        # Update the bot's nickname in all servers
    for guild in bot.guilds:
        me = guild.me
        await me.edit(nick="Llama Bot")

def create_context(guild_id):
    # If this server doesn't have a context yet, create one
    if guild_id not in contexts:
        user, assistant = prompt_format_dict[model_str]
        contexts[guild_id] = ChatContext(INITIAL_CONTEXT, enc, 2048, user, assistant)

@bot.event
async def on_message(message):
    if message.channel.name != 'llm':
        return

    if message.author == bot.user:
        return

    if message.content[0] == '!':
       await bot.process_commands(message)

    else:
        # Get the server ID
        guild_id = message.guild.id

        create_context(guild_id)

        # Add the message to the context
        contexts[guild_id].add_user_input(message.content)

        # Get the context for this server
        async with message.channel.typing():
            llm_str = contexts[guild_id].get_context_str()
            user, assistant = prompt_format_dict[model_str]
            stream = llm(
                llm_str,
                max_tokens=1024,
                repeat_penalty= 1.1,
                temperature=0.7,
                stop=[user.strip()],
                stream=True,
            )
        
        bot_message = None
        string = ""
        start_time = time.time()
        for outputs in stream:
            string += outputs["choices"][0]['text']

            if bot_message is None and len(string.strip(" ")) > 0:
                bot_message = await message.channel.send(string)

            if (time.time() - start_time) >= STREAM_UPDATE_PERIOD and bot_message is not None:
                start_time = time.time()
                await bot_message.edit(content=string)

        if bot_message is not None:
            await bot_message.edit(content=string)
        else:
            bot_message = await message.channel.send("*stares at you akwardly...")

        contexts[guild_id].add_assistant_output(string)


@bot.command()
async def reset_chat(ctx):
        
    if ctx.channel.name != 'llm':
        return
    
    guild_id = ctx.guild.id
    # If this server doesn't have a context yet, create one
    create_context(guild_id)

    contexts[guild_id].reset_context()
    await ctx.channel.send("RESET CHAT HISTORY (THE BOT HAS FORGETTON THE CONVERSATION SO FAR)")


@bot.command()
async def set_initial_context(ctx, *, initial_context):
        
    if ctx.channel.name != 'llm':
        return
    
    guild_id = ctx.guild.id
    # If this server doesn't have a context yet, create one
    create_context(guild_id)

    contexts[guild_id].set_initial_context(initial_context)
    await ctx.channel.send("THE BOT'S PERSONALITY AND THE CHAT IS NOW INFLUENCED BY TO YOUR TEXT: \n\n" + initial_context)

    
@bot.command()
async def switch_model(ctx, model):
        
    global llm, model_str                              # Ensure we are accessing the correct global variables

    if ctx.channel.name != 'llm':
        return
    
    guild_id = ctx.guild.id
    # If this server doesn't have a context yet, create one
    create_context(guild_id)

    model_str = model_dict[int(model)]
    user, assistant = prompt_format_dict[model_str]
    contexts[guild_id].set_prompt_style(user, assistant)
    del llm
    llm = Llama(model_path=model_str, n_threads=14, n_gpu_layers=43, seed=-1, n_ctx=2048)

    await ctx.channel.send("THE BOT IS NOW RUNNING ON A DIFFERENT MODEL: " + model_str)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        # Report mistyped command
        await ctx.send("Invalid command. Please check your command and try again.")
    else:
        # Handle other errors
        print(f'Error occurred: {error}')


token = os.getenv('DISCORD_TOKEN')

# Make sure the token is not None before running the bot
if token:
    bot.run(token)
else:
    print("DISCORD_TOKEN environment variable is not set.")