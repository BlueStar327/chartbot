from flask import Flask, request, session, render_template,jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (AgentTokenBufferMemory, )
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
import os
import openai   
from dotenv import find_dotenv, load_dotenv
from flask_cors import CORS

from langchain.agents import AgentExecutor, BaseMultiActionAgent, Tool

# Generate a random hexadecimal string, 24 bytes long
secret_key = os.urandom(24).hex()

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")



app = Flask(__name__)
app.secret_key = secret_key # Set a secret key for session management
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Authorization", "Content-Type"]}})

@app.route('/', methods=['GET'])
def index():
    return "Welcome to bot."




def signal():
    embeddings = OpenAIEmbeddings()
    ### load vector DB embeddings
    vectordb = FAISS.load_local(
        './faiss_index_site',
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever(search_kwargs = {"k": 5, "search_type" : "similarity"})
    return retriever



def gpt_ai_response(query,history):

    msg = []

    # Process history messages
    for message_data in history:
        role = message_data['role']
        content = message_data['content']
        if role == 'assistant':
            msg.append(AIMessage(content=content))
        elif role == 'user':
            msg.append(HumanMessage(content=content))

    Our_data = create_retriever_tool(
    signal(),
    "get_our_data",
    "Always refer it, it Searches and returns documents regarding the business.",
    )



    tools = [Our_data]

    llm = ChatOpenAI(temperature=0.2, streaming=True, model="gpt-4-1106-preview")
    message = SystemMessage(content=(
"""
    You are an AI Assistant and the customer service agent for the company Positive energy, you are one of the top energy suppliers. This is the company website https://pozitive.energy/.
    Make sure you are friendly and respectful.
    User can contact us at  Customer Care on 0333 370 9900 or customercare@pozitive.energy.
{Context}

Client:{question}

AI Assistant:

"""
    ))
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    memory = AgentTokenBufferMemory(llm=llm,return_messages=True)
    starter_message = "Your AI Assistant, feel free to ask anything about our company Positive Energy."
    # msg=[]
    msg.append(AIMessage(content=starter_message))
    # print(msg)
    response = agent_executor(
            {
                "input": query,
                "history": msg,
            },
            include_run_info=True,
        )

    memory.chat_memory.add_message(response["output"])
    return response['output']

@app.route('/ai_response', methods=['POST'])
def ai_response_endpoint():
    # Get the message from the POST request
    data = request.json
    query = data.get('human_input')
    history= data.get('history')
    print(query)
    answer=gpt_ai_response(query,history)
    return jsonify(answer)


if __name__ == '__main__':
    app.run(debug=True)
