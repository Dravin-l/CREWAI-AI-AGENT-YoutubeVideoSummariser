from crewai import Agent,LLM
from tools import yt_tool

from dotenv import load_dotenv

load_dotenv()



import os
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

llm = LLM(
    model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
    base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key=HUGGINGFACE_API_KEY
)


## Create a senior blog content researcher

blog_researcher=Agent(
    role='Blog Researcher from Youtube Videos',
    goal='get the relevant video transcription for the topic {topic} from the provided Yt channel',
    verboe=True,
    memory=True,
    backstory=(
       "Expert in understanding videos in AI Data Science , MAchine Learning And GEN AI and providing suggestion" 
    ),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=True
)

## creating a senior blog writer agent with YT tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False


)