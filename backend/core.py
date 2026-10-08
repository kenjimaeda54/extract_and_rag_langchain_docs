import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

open_router_api_key = os.environ.get("OPEN_ROUTER_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")




embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name="lang-docs-rag",
    pinecone_api_key=pinecone_api_key
)
model = init_chat_model(
    model="openai/gpt-4o-mini",
    model_provider="openai",
    api_key=open_router_api_key,
    base_url="https://openrouter.ai/api/v1",
)


SYSTEM_PROMPT = (
    "You answer questions about LangChain documentation. "
    "Always call the retrieval tool before answering and cite the sources you use. "
    "If the answer is not in the retrieved documentation, say so."
)



#vai retornar o conteúdo do documento e o artefato (vetor) correspondente
#           retrieve_context()
#                   │
#                   ▼
#           ┌───────────────┐
#           │  dois valores │
#           └───────┬───────┘
#                   │
#        ┌──────────┴──────────┐
#        ▼                     ▼
#   serialized                docs
#    (string)           (lista Document)
#       │                     │
#       │                     │
#       ▼                     ▼
#     CONTEXT           ARTIFACT
#    para  o LLM        dados originais

@tool(
    response_format="content_and_artifact"
)
def retrieve_context(query: str):
    """
    Retrieve relevant LangChain documentation for a query.
    """
    docs=  vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source','Unknow')}\n\nContent: {doc.page_content}"
           for doc in docs
        ]
    )
    # Return both serialized content and raw documents
    # com o raw consigo mostrar de onde vem o documento, avaliar o RAG e debugar
    # depois reaproveito o docs para exibir no front-end, por exemplo, no Streamlit
    return serialized,docs


#agente fora cria apenas uma vez
#se estiver dentro do run_llm, toda vez que a função for chamada, o agente será recriado, o que é ineficiente
agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

def run_llm(query: str) -> Dict[str,Any]:
    """
       Run the RAG pipeline to answer a query using retrieved documentation.

       Args:
           query: The user's question

       Returns:
           Dictionary containing:
               - answer: The generated answer
               - context: List of retrieved documents
    """

    #regra de role
    #system	você, o desenvolvedor	instruções de comportamento ("responda só com base na documentação")
    # user	a pessoa usando o app	perguntas e pedidos
    # assistant	o modelo	respostas dele (útil para passar o histórico)
    # tool	uma ferramenta	resultado de uma tool, como o retrieve_context
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    #abaixo nao ira remover os duplicados
    #para chegar no codigo abaixo pode ir debugando aos poucos

    # python
    # # Ver o tipo de cada coisa
    # for msg in response["messages"]:
    #     print(type(msg).__name__)
    # # HumanMessage, AIMessage, ToolMessage, AIMessage
    #
    # # Ver os campos de um objeto
    # print(response["messages"][2])
    #
    # # Ver o que um objeto tem disponível
    # print(dir(response["messages"][2]))
    #
    # # Ver o tipo do artifact
    # print(type(response["messages"][2].artifact))
    # # <class 'list'>
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
           if isinstance(message.artifact,list):
                context_docs.extend(message.artifact)
                # ele esta adicionado o Document que a tool recuperou, mas nao esta removendo os duplicados
                #     Document(page_content="### Copy thread...", metadata={"source": "https://docs.langchain.com/langsmith/use-threads"}),
                # esta adiconado o segundo valor da tool    return serialized,docs
                # retornar o docs ajuda:
                # 1. Mostrar de onde veio a resposta.
                # 2. Avaliar o RAG
                # 3. Debug.



    #retornar a ultima mensagem do modelo (a resposta final)
    # e os documentos de contexto
    return {
        "answer": response["messages"][-1].content,
        "context": context_docs
    }



if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)
