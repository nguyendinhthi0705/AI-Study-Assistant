import os
import boto3, json
from dotenv import load_dotenv
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
load_dotenv()

def call_claude_sonet_stream(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0, 
        "top_k": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    bedrock = boto3.client(service_name="bedrock-runtime")  
    response = bedrock.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    stream = response['body']
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                 delta = json.loads(chunk.get('bytes').decode()).get("delta")
                 if delta:
                     yield delta.get("text")    
    
def rewrite_document(input_text): 
    prompt = """Your name is good writer. You need to rewrite content: 
        \n\nHuman: here is the content
        <text>""" + str(input_text) + """</text>
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)


def summary_stream(input_text):     
    prompt = f"""Based on the provided context, create summary the lecture
        \n\nHuman: here is the content
        <text>""" + str(input_text) + """</text>
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)

def query_document(question, docs): 
    prompt = """Human: here is the content:
        <text>""" + str(docs) + """</text>
        Question: """ + question + """ 
    \n\nAssistant: """

    return call_claude_sonet_stream(prompt)

def create_questions(input_text): 
    system_prompt = """You are an expert in creating high-quality multiple-choice quesitons and answer pairs 
    based on a given context. Based on the given context (e.g a passage, a paragraph, or a set of information), you should:
    1. Come up with thought-provoking multiple-choice questions that assess the reader's understanding of the context. 
    2. The questions should be clear and concise.
    3. The answer options should be logical and relevant to the context.

    The multiple-choice questions and answer pairs should be in a bulleted list: 
        1) Question: 

        A) Option 1

        B) Option 2 

        C) Option 3 

        Answer: A) Option 1 

         
    Continue with additional questions and answer pairs as needed.

    MAKE SURE TO INCLUDE THE FULL CORRECT ANSWER AT THE END, NO EXPLANATION NEEDED:"""
    
    prompt = f"""{system_prompt}. Based on the provided context, create 10 multiple-choice questions and answer pairs
        \n\nHuman: here is the content
        <text>""" + str(input_text) + """</text>
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)

def suggest_writing_document(input_text): 
    prompt = """Your name is good writer. You need to suggest and correct mistake in the essay: 
        \n\nHuman: here is the content
        <text>""" + str(input_text) + """</text>
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)

def search(question, callback): 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="PQVFC2WMPJ",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}},

    )

    model_kwargs_claude = {"max_tokens": 1000}
    llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0"
                      , model_kwargs=model_kwargs_claude
                      , streaming=True
                      , callbacks=[callback])

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return chain.invoke(question)

def search_new(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime")  
    
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id = "PQVFC2WMPJ", 
        top_k = 3,
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": 5, 
                'overrideSearchType': "SEMANTIC"
            }
        }
    )
    
    retrieved_docs = retriever.get_relevant_documents(prompt + " 2024")
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    system_prompt = """
    You are an advanced AI financial advisor with extensive market knowledge and analytical capabilities. Your name is CRobo Advisor. The current date is July 2024. Your role is to assist traders and investors with stock analysis, market insights, and trading strategies. Please adhere to the following guidelines:
    1. Expertise: Demonstrate deep understanding of financial markets, trading strategies, technical analysis, and market structures.
    2. Analysis: Provide thorough, data-driven analysis of market trends, specific stocks, or trading strategies as requested.
    3. Explanations: Offer clear, comprehensive explanations suitable for traders of all experience levels. Break down complex concepts when necessary.
    4. Asset Evaluation: When asked, use your knowledge to identify potential trading assets. Provide a detailed list with supporting data and rationale for each recommendation.
    5. Current Information: Ensure all advice and analysis is based on the most up-to-date market information available to you. If you need to access real-time data, inform the user and proceed to retrieve the latest information.
    6. Honesty: If you're unsure about something or don't have the necessary information, clearly state this. Do not provide speculative or potentially misleading information.
    7. Language: Provide all responses in Vietnamese.
    8. Adaptability: Tailor your responses to the specific needs and questions of each user, whether they're seeking general market insights or detailed analysis of particular stocks or strategies.
    9. Markdown Format: Provide all responses in Markdown format, highlighting key points using bold text.
    """
    query = f"""Human: {system_prompt}. Based on the provided context, provide the answer to the following question:
    <context>{context}</context>
    <question>{prompt} </question>
    Remember, while you can offer analysis and insights, you should not make definitive predictions about future market movements or provide personalized financial advice.
    Assistant: 
    """
 
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "top_p": 1,
    }
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

    response = bedrock.invoke_model_with_response_stream(
        body=json.dumps(prompt_config),
        modelId=modelId,
        accept="application/json", 
        contentType="application/json"
    )

    stream = response['body']
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if 'delta' in chunk_obj:
                    delta_obj = chunk_obj.get('delta', None)
                    if delta_obj:
                        text = delta_obj.get('text', None)
                        yield text
