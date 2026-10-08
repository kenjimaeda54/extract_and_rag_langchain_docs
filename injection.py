import hashlib
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

tavily_crawl = TavilyCrawl()

# Ferramenta que extrai o conteúdo de uma ou mais URLs.
# Retorna um dicionário com "results" (url + raw_content) e "failed_results".
tavily_extract = TavilyExtract()


# max_depth
#
# É a profundidade: quantos "cliques" de distância da URL inicial o mapeamento pode chegar.
#
# max_depth=1: só as páginas linkadas diretamente na página inicial.
# max_depth=2: essas páginas + as páginas que elas linkam.
# max_depth=3: mais um nível além disso.
#
# Exemplo: home → /blog (nível 1) → /blog/post-1 (nível 2) → /blog/post-1/comentarios (nível 3).
#
# max_breadth
#
# É a largura: quantos links o crawler segue por página (em cada nível).
# Com max_breadth=20, em cada página ele pega no máximo 20 links e ignora o resto.
# limit
#
# É o teto total de páginas processadas. Mesmo que depth e breadth permitam mais, ele para em 1000.
# Como chegar nesses valores
#
# Não existe número mágico, depende do site e do objetivo. Uma forma prática:
#
# Comece pequeno: max_depth=1, max_breadth=10, limit=50. Veja o que volta.
# Aumente gradualmente a profundidade se estiver faltando conteúdo importante.
# Aumente a largura se páginas com muitos links (ex: listagens) estiverem sendo cortadas.
# Ajuste o limit para controlar custo e tempo.
#
# Regras gerais:
#
# Site de documentação (estrutura em árvore): depth 2-3, breadth 20-50.
# Blog ou site pequeno: depth 1-2, breadth 10-20.
# Site grande/e-commerce: depth baixo (1-2) com limit alto, senão o número de páginas explode.
# Ferramenta que mapeia o site e retorna apenas a lista de URLs (sem conteúdo).

tavily_map = TavilyMap(
   max_depth=3,
   max_breadth=20,
   limit=1000
)

#Diferneça entre TavilyExtract e TavilyCrawl:

#TavilyExtract: você entrega as URLs e ele só extrai o conteúdo delas. Não segue links nem descobre nada.
#TavilyCrawl: você entrega uma URL inicial e ele navega sozinho pelo site,
# seguindo links (respeitando max_depth, max_breadth e limit), e extrai o conteúdo de cada página que encontra.

# Ferramenta que percorre o site e extrai o conteúdo de cada página.
# Retorna um dicionário com "base_url" e "results" (lista de url + raw_content).

pinecone_vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name="lang-docs-rag",
    pinecone_api_key=PINECONE_API_KEY

)

@traceable(name="injection")
def main():
    ##existe abaixo a opção instructions
    ##e util quando queremos um rasp mais especifico, por exemplo, extrair apenas os títulos de uma página
    #mas pode gerar erros se a instrução não for clara ou se a página tiver um layout complexo
    results = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5,
        "instructions": "advanced"
    })
    all_documents = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}
        )
        for result in results["results"]
        if  result.get("raw_content")
    ]# so pega conteudos que existem por isso if result.get("raw_content") haviam conteudos vindo None

    recursive_character = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = recursive_character.split_documents(all_documents)
    ids = [
        hashlib.sha256(f"{c.metadata['source']}::{c.page_content}".encode()).hexdigest()
        for c in chunks
    ]
    pinecone_vectorstore.add_documents(chunks, ids=ids,batch_size=100)




    #para enviar ao langsmith
    return {
        "documents": all_documents,
        "num_documents": len(all_documents)
    }



if __name__ == "__main__":
    main()
