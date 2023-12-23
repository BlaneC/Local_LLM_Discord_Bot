import discord
from discord.ext import commands
import json
import aiohttp
import asyncio
import aiofiles
import os
from dotenv import load_dotenv
import subprocess
import re
import base64


# Load the environment variables from .env file
load_dotenv()

model = "dolphin-mixtral:8x7b-v2.5-q4_0"  # Update this for whatever model you wish to use
llava_model = "llava:13b"

BOT_NAME = "Dolphin Mixtral Bot"
STREAM_UPDATE_PERIOD = 0.5  # Time in seconds between message updates
MAX_MESSAGE_LENGTH = 1500  # Maximum Discord message length before sending a new message
SYSTEM_MESSAGE = """I am a helpful Discord bot, here to answer your questions, provide information, and make your experience on Discord more enjoyable.
                Should you ever feel that our conversation has strayed from the topic at hand or that you would like to start fresh with a new query, 
                simply type !clear_context in any channel where I am present. This command will reset my context window, allowing us to begin anew with a clean slate.
                I am programmed to be patient and understanding, so feel free to ask me anything, no matter how simple or complex the question may be. 
                My primary goal is to ensure that our interaction is informative, engaging, and enjoyable for all parties involved. Let's work together to broaden our knowledge and perspectives!
                """
TEMP_IMAGE_DIR = r"C:\Users\bcypu\Documents\Projects\Discord-Bot\temp"
DOCKER_IMAGE_DIR = "/tmp/temp_images"   # Path inside the Docker container

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize an empty dictionary to store contexts
contexts = {}
content_lock = asyncio.Lock()

def start_docker_container():
    try:
        # Create the temporary image directory if it doesn't exist
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

        subprocess.run([
            "docker", "run", "-d", "--gpus=all",
            "-v", "ollama:/root/.ollama",
            "-p", "11434:11434", "--name", "ollama", "ollama/ollama"
        ], check=True)
        # Run the Docker container with the temporary image directory volume
        subprocess.run([
            "docker", "run", "-d", "--gpus=all",
            "-e", "OLLAMA_HOST=0.0.0.0:11435"
            "-v", "ollama:/root/.ollama",
            "-p", "11435:11435", "--name", "ollama_lava", "ollama/ollama"
        ], check=True)

        # Execute the program in the container (non-interactive mode)
        subprocess.run(["docker", "exec", "ollama", "ollama", "run", model], check=True)
        subprocess.run(["docker", "exec", "ollama_lava", "ollama", "run", "llava:13b"], check=True)

        print("Docker container started and program executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while starting the Docker container: {e}")


async def update_message_periodically(bot_message, shared_content):
    while not shared_content['done']:
        await asyncio.sleep(STREAM_UPDATE_PERIOD)
        async with content_lock:
            current_content = shared_content['content']

            if len(current_content) > MAX_MESSAGE_LENGTH:
                # Find the last word boundary before MAX_MESSAGE_LENGTH
                split_index = current_content.rfind(' ', 0, MAX_MESSAGE_LENGTH)
                if split_index == -1:
                    # If no space is found, use MAX_MESSAGE_LENGTH
                    split_index = MAX_MESSAGE_LENGTH

                message_to_send = current_content[:split_index]
                remaining_content = current_content[split_index:].strip()
                shared_content['content'] = remaining_content
                
                await bot_message.edit(content=message_to_send)
                bot_message = await bot_message.channel.send(remaining_content)
            elif current_content:
                # Edit the existing message with the current content
                await bot_message.edit(content=current_content)

async def stream_chat(messages, shared_content):
    message_buff = ""
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/api/chat", json={"model": model, "messages": messages, "system": SYSTEM_MESSAGE, "stream": True}) as response:
            async for line in response.content:
                body = json.loads(line.decode('utf-8'))
                if "error" in body:
                    raise Exception(body["error"])
                else:
                    message_buff += body.get("message", {}).get("content", "")
                async with content_lock:
                    if body.get("done") is False:
                        shared_content['content'] += message_buff
                        message_buff = ""
                    if body.get("done", False):
                        shared_content['done'] = True
                        break

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    for guild in bot.guilds:
        me = guild.me
        await me.edit(nick=BOT_NAME)

def create_context(guild_id):
    if guild_id not in contexts:
        contexts[guild_id] = []

@bot.command(name='set_system_message')
async def set_system_message(ctx, *, system_message: str):
    global current_system_message
    current_system_message = system_message
    await ctx.send(f"System message updated to: {system_message}")

@bot.command(name='clear_context')
async def clear_context(ctx):
    guild_id = ctx.guild.id
    if guild_id in contexts:
        contexts[guild_id] = []
        await ctx.send("Context cleared.")
    else:
        await ctx.send("No context to clear.")

async def download_image(url, session, download_path):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                # Extract the filename and sanitize it
                filename = re.sub(r'[\\/*?:"<>|]', "", os.path.basename(url).split('?')[0])
                filepath = os.path.join(download_path, filename)
                
                async with aiofiles.open(filepath, 'wb') as file:
                    await file.write(await response.read())
                return filepath
    except Exception as e:
        print(f"Error downloading image: {e}")
    return None

async def get_image_descriptions(images):
    descriptions = []
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        for image_url in images:
            image_path = await download_image(image_url, session, TEMP_IMAGE_DIR)
            if image_path:
                # Read the image and encode it to base64
                async with aiofiles.open(image_path, 'rb') as image_file:
                    image_data = await image_file.read()
                    base64_encoded_image = base64.b64encode(image_data).decode('utf-8')

                payload = {
                    "model": llava_model,
                    "prompt": "Describe in detail what is in this picture. Preface your description with a title for the picture. Write all text that appears in the image seperately from your main description, word for word",
                    "stream": False,
                    "images": [base64_encoded_image]
                }

                async with session.post("http://localhost:11435/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        description = result.get("response", "")
                        if description:
                            descriptions.append(description)
                
                # Delete the image from the host after processing
                os.remove(image_path)

    return descriptions

@bot.event
async def on_message(message):
    if message.channel.name != 'llm':
        return

    if message.author == bot.user:
        return

    if message.content.startswith('!'):
        await bot.process_commands(message)
        return

    guild_id = message.guild.id
    create_context(guild_id)

    image_descriptions = []
    bot_message = await message.channel.send("Thinking...")

    if message.attachments and message.channel.permissions_for(message.guild.me).attach_files:
        await bot_message.edit(content="Looking at images...")
        # Filter attachments for image file types
        images_to_process = [attachment.url for attachment in message.attachments if any(attachment.filename.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.svg'))]
        image_descriptions = await get_image_descriptions(images_to_process)

    elif not message.channel.permissions_for(message.guild.me).attach_files:
        await message.channel.send("I don't have permission to analyze images.")

    async with content_lock:
        image_description_text = '\nDescription of images attached: \n'.join(image_descriptions)
        contexts[guild_id].append({"role": "user", "content": message.content + image_description_text})

    await bot_message.edit(content="Thinking...")
    shared_content = {'content': '', 'done': False}

    # Run streaming and message updating concurrently
    await asyncio.gather(
        stream_chat(contexts[guild_id], shared_content),
        update_message_periodically(bot_message, shared_content)
    )
    async with content_lock:
        contexts[guild_id].append({"role": "assistant", "content": shared_content["content"]})



# Call the function to start the Docker container
start_docker_container()

# Start the Discord bot
token = os.getenv('DISCORD_TOKEN')
if token:
    bot.run(token)
else:
    print("DISCORD_TOKEN environment variable is not set.")
