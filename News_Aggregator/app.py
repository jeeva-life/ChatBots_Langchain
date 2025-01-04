import streamlit as st
import itertools
from itertools import zip_longest
import emoji

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_community.agent_toolkits import load_tools
from langchain_community.utilities import SerpAPIWrapper
#from langchain.agents.load_tools import load_tools
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv
import os

load_dotenv()

open_api_key= os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")


def get_text():
    input_text = st.sidebar.text_input("Input: ", key='input')
    if st.sidebar.button('Send'):
        return input_text
    return None

def get_history(history_list):
    history=''
    for message in history_list:
        if message['role'] == 'user':
            history += 'input:' + message['content']
        elif message['role'] == 'assistant':
            history += 'output:' + message['content']
    return history 

def get_response(history, user_message, temperature = 0):
    DEFAULT_TEMPLATE ="""As an AI-powered digital journalist, you have an expertise in comprehending, summarizing, and delivering information sourced from reputable news outlets. You maintain a firm grasp on current trends and hot news topics, providing users with verified and unbiased insights in a conversational style. The user will interact with you to learn about the latest headlines, getting informed about trending topics and stories they are interested in. In every interaction, your focus is to provide information that is accurate, timely, and clear.
        It follows the previous conversation to do so.

        Relevant pieces of previous conversation:
        {context},

        Useful news information from Web:
        {web_knowledge},

        Current conversation:
        Human: {input}
        News Journalist:"""
    prompt = PromptTemplate(input_variables=['context', 'web_knowledge','input'], template=DEFAULT_TEMPLATE)

    chat_gpt = ChatOpenAI(temperature=temperature, model_name = 'gpt-3.5-turbo', openai_api_key=open_api_key)

    search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    from langchain.tools import Tool

    tools = [Tool(
    name="Search",
    func=search.run,
    description="Use this tool to search the web for up-to-date information on job postings."
    )
    ]

    agent = initialize_agent(tools,chat_gpt, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
            handle_parsing_errors='Check output and make sure it confirms')

    web_knowledge = agent.run('Fetch detailed analysis without summarizing from news articles regarding' + user_message)

    conversation = LLMChain(
        llm = chat_gpt,
        prompt=prompt,
        verbose=True
    )
    response = conversation.predict(context=history, input=user_message, web_knowledge=web_knowledge)

    return response


st.title("News Aggregator")

user_input = get_text()

if "past" not in st.session_state:
    st.session_state["past"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = []


if user_input:
    user_history = list(st.session_state['past'])
    bot_history = list(st.session_state['generated'])

    combined_history = []

    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)

    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            st.markdown(emoji.emojize(f":speech_balloon: **User**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":Robot: **Assistant**: {st.session_state['generated'][i]}"))
