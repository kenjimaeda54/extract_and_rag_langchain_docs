import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from torch.distributed._shard.sharded_tensor import metadata

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pinecone_vector = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    index_name="lang-docs-rag",
    embedding=embeddings
)

# Ferramenta que extrai o conteúdo de uma ou mais URLs.
# Retorna um dicionário com "results" (url + raw_content) e "failed_results".
travail_extract = TavilyExtract()

# Ferramenta que mapeia o site e retorna apenas a lista de URLs (sem conteúdo).
travail_map = TavilyMap(
    max_breadth=20,
    max_depth=3,
    max_pages=1000,
)

#Diferneça entre TavilyExtract e TavilyCrawl:

#TavilyExtract: você entrega as URLs e ele só extrai o conteúdo delas. Não segue links nem descobre nada.
#TavilyCrawl: você entrega uma URL inicial e ele navega sozinho pelo site,
# seguindo links (respeitando max_depth, max_breadth e limit), e extrai o conteúdo de cada página que encontra.

# Ferramenta que percorre o site e extrai o conteúdo de cada página.
# Retorna um dicionário com "base_url" e "results" (lista de url + raw_content).
travail_crawl = TavilyCrawl()



def main():
    ##existe abaixo a opção instructions
    ##e util quando queremos um rasp mais especifico, por exemplo, extrair apenas os títulos de uma página
    #mas pode gerar erros se a instrução não for clara ou se a página tiver um layout complexo
    res = travail_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5,
        "extract_depth": "advanced",
    })
    all_docs = [
                Document(page_content=result["raw_content"],
                         metadata={"source": result["url"]})
                for result in res["results"]
                if result.get("raw_content")
    ] # so pega conteudos que existem por isso if result.get("raw_content") haviam conteudos vindo None
    print(f"Number of documents extracted: {len(all_docs)}")



if __name__ == "__main__":
    main()
