# LangChain Docs RAG (Agentic RAG)

Chatbot de perguntas sobre a documentação do LangChain. Usa **Agentic RAG**: um agente com tool-calling decide quando consultar a base vetorial antes de responder.

---

## Sumário

1. [Visão geral do projeto](#visão-geral-do-projeto)
2. [Tipos de RAG: Clássico, Agentic e Hybrid](#tipos-de-rag-clássico-agentic-e-hybrid)
3. [Por que este projeto é Agentic RAG](#por-que-este-projeto-é-agentic-rag)
4. [Arquitetura do sistema](#arquitetura-do-sistema)
5. [Frontend com Streamlit](#frontend-com-streamlit)
6. [Como os Sources são exibidos](#como-os-sources-são-exibidos)
7. [Decisões de implementação (para consulta futura)](#decisões-de-implementação-para-consulta-futura)
8. [Pipeline de injeção (indexação)](#pipeline-de-injeção-indexação)
9. [Como rodar](#como-rodar)

---

## Visão geral do projeto

| Parte | Tecnologia |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | Pinecone (`lang-docs-rag`) |
| LLM | GPT-4o-mini via OpenRouter |
| Agente | `langchain.agents.create_agent` + tool `retrieve_context` |
| UI | Streamlit (chat) |
| Ingestão | Tavily (crawl) + RecursiveCharacterTextSplitter |

**Fluxo resumido:**

```
Pergunta do usuário
       │
       ▼
Agente (LLM) decide chamar a tool retrieve_context
       │
       ▼
similarity_search no Pinecone → top-4 chunks
       │
       ├─ serialized  → contexto que o LLM vê
       └─ docs        → artifact (Document brutos com metadata)
       │
       ▼
LLM gera a resposta usando o contexto
       │
       ▼
UI extrai sources do artifact e mostra no expander
```

---

## Tipos de RAG: Clássico, Agentic e Hybrid

### 1. RAG Clássico (pipeline fixo)

É o RAG "de tutorial": um fluxo linear, sem decisão do modelo.

```
query → embed → similarity_search → injetar no prompt → gerar resposta
```

**Características:**
- A recuperação **sempre acontece**, antes de gerar
- O modelo **não decide** se quer (ou precisa) buscar
- Um único passo de retrieval, fixo
- Não há loop de raciocínio nem chamada condicional de ferramentas

**Exemplo simplificado (pseudo-código):**

```python
docs = vector_store.similarity_search(query, k=4)
context = "\n".join(d.page_content for d in docs)
answer = llm.invoke(f"Responda usando:\n{context}\n\n{query}")
```

**Quando usar:** perguntas diretas, domínio bem coberto, latência baixa.

---

### 2. Agentic RAG (o que este projeto usa)

O LLM vira um **agente** com ferramentas (tools). Ele decide **se, quando e como** recuperar informação antes de responder.

```
query
  │
  ▼
Agente (LLM com tools)
  │
  ├─ (opcional) decide chamar retrieve_context
  │         │
  │         ▼
  │   similarity_search → contexto
  │         │
  │         ▼
  │   LLM avalia o resultado
  │         │
  │         └─ se insuficiente, pode chamar de novo (loop)
  │
  ▼
resposta final (com sources)
```

**Características que definem Agentic RAG:**

| Critério | Presente? | Como aparece aqui |
|---|---|---|
| Uso de agent framework | Sim | `create_agent(...)` do LangChain |
| Tool-calling (o modelo chama a tool) | Sim | `retrieve_context` com `@tool` |
| O modelo **decide** quando buscar | Sim | o SYSTEM_PROMPT manda consultar antes, mas quem dispara a tool é o agente |
| Possibilidade de loop / múltiplas chamadas | Sim (em potencial) | o agente pode reavaliar e chamar a tool de novo |
| Mais de uma ferramenta / roteamento | Não (nesta versão) | apenas 1 tool de retrieval |
| Raciocínio entre tool calls | Básico | o agente vê o resultado e gera a resposta |

**Na prática, este projeto é um Agentic RAG "single-tool" / minimalista:**
- Uma única tool de retrieval
- Sem query rewriting, re-ranking, self-critique ou plano multi-step
- O modelo decide a chamada, mas o pipeline de ferramentas é simples

RAG clássico faria retrieve sempre, na mão do código. Aqui o **agente** (o LLM) é quem orquestra a chamada da tool.

---

### 3. Hybrid RAG

Híbrido = combina **mais de uma estratégia de recuperação** (ou de geração) no mesmo pipeline.

Exemplos comuns de hibridização:

| Combinação | O que junta |
|---|---|
| **Dense + Sparse** | embeddings (semântico) + BM25/keyword (léxico) |
| **Multi-index** | buscas em bases diferentes (docs + código + FAQ) |
| **Retrieve + web search** | base interna + busca na internet |
| **Multi-agent** | um agente busca, outro gera, outro revisa |
| **RAG + Graph** | vetores + grafo de conhecimento |

**Este projeto NÃO é Hybrid RAG:**
- Só existe **um** índice (Pinecone `lang-docs-rag`)
- Só existe **um** tipo de busca: `similarity_search` (denso/embedding)
- Não tem BM25, keyword search, web search nem grafo
- Não tem múltiplos agentes ou etapas especializadas

Se no futuro você adicionar busca híbrida (dense + sparse) no Pinecone ou uma tool extra de web search aí sim o projeto passaria a ser (ou a conter) Hybrid RAG.

---

### Comparativo rápido

| | RAG Clássico | Agentic RAG | Hybrid RAG |
|---|---|---|---|
| Quem decide buscar? | o código (sempre) | o agente/LLM | o código ou agentes |
| Recuperação | sempre, fixa | condicional / orquestrada | múltiplas estratégias |
| Ferramentas | nenhuma (inline) | sim (`@tool`) | várias tools/índices |
| Complexidade | baixa | média | média-alta |
| Este projeto | não | **sim** | não |

---

## Por que este projeto é Agentic RAG

Checklist baseado no código real:

1. **`create_agent`** — o orquestrador é um agente, não uma função linear.
   ```python
   agent = create_agent(model=model, tools=[retrieve_context], system_prompt=SYSTEM_PROMPT)
   ```

2. **Tool com `@tool`** — a recuperação é uma **ferramenta**, não um passo fixo.
   ```python
   @tool(response_format="content_and_artifact")
   def retrieve_context(query: str): ...
   ```

3. **`agent.invoke` em vez de `similarity_search` direto no fluxo principal** — o LLM decide se chama a tool.
   ```python
   response = agent.invoke({"messages": [{"role": "user", "content": query}]})
   ```

4. **SYSTEM_PROMPT orienta o comportamento do agente** — instrução de procedimento, não código rígido.
   ```python
   "Always call the retrieval tool before answering and cite the sources you use."
   ```

5. **Loop implícito possível** — se a resposta do tool for fraca, o agente pode chamar `retrieve_context` outra vez com outra query (o framework suporta; esta versão usa 1 chamada em casos simples).

**Em uma frase:** RAG clássico = código busca e o modelo só responde; Agentic RAG = o modelo é o agente e a recuperação é mais uma das ferramentas que ele pode usar.

---

## Arquitetura do sistema

```
┌─────────────────────────────────────────────────────────┐
│  injection.py  (uma vez / quando precisa reindexar)     │
│  TavilyCrawl → Document(source=url) → chunks → Pinecone │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  backend/core.py   (runtime)                            │
│                                                         │
│  embeddings (all-MiniLM-L6-v2)                          │
│  vector_store (Pinecone lang-docs-rag)                  │
│  model (GPT-4o-mini via OpenRouter)                     │
│                                                         │
│  @tool retrieve_context                                  │
│    similarity_search(query, k=4)                        │
│    return serialized, docs   ← content_and_artifact    │
│                                                         │
│  agent = create_agent(tools=[retrieve_context])         │
│                                                         │
│  run_llm(query) → {answer, context}                     │
│    - extrai artifact das ToolMessage                    │
│    - retorna lista de Document para a UI                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  main.py   (Streamlit UI)                               │
│  - chat com session_state                               │
│  - run_llm(prompt)                                      │
│  - _format_sources(context) → lista de URLs             │
│  - expander "Sources" no chat                           │
└─────────────────────────────────────────────────────────┘
```

---

## Como os Sources são exibidos

A cadeia completa em 3 camadas:

### 1. Tool devolve dois valores

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}"
        for doc in docs
    )
    return serialized, docs
```

| Valor retornado | Quem recebe | Para quê |
|---|---|---|
| `serialized` (string) | o LLM | contexto da resposta (o modelo não "lê" objetos Document) |
| `docs` (lista de `Document`) | a aplicação via `.artifact` | mostrar sources, debugar, avaliar o RAG |

O `Document` carrega `metadata["source"]` com a URL da docs LangChain. Sem esse artifact, o LLM responderia bem, mas você **perderia o rastro de onde veio cada trecho**.

### 2. `run_llm` extrai os artifacts

Depois do `agent.invoke`, o código percorre as mensagens, acha as `ToolMessage`, lê o `.artifact` e monta `context_docs`.

> **Nota:** nesta versão o código faz `context_docs.extend(message.artifact)` — se a tool for chamada 2x, o mesmo chunk pode aparecer duas vezes. Deduplicar (ex.: `sha256(source + page_content)`) é um improvement natural para o futuro.

### 3. Streamlit formata e exibe

`_format_sources` extrai `metadata["source"]` de cada doc e o UI mostra num `st.expander("Sources")`.

**Resumo:** o artifact é a "prova documental" que viaja junto com a resposta do agente — é isso que torna a exibição de Sources possível.

---

## Frontend com Streamlit

A UI é um chat simples em `main.py`. Streamlit roda o script de cima para baixo a cada interação (rerun) — por isso o estado precisa ficar em `st.session_state`.

### Como o chat foi construído

| Peça | Código | Papel |
|---|---|---|
| Config da página | `st.set_page_config(page_title=..., layout="centered")` | título da aba e layout |
| Sidebar | `with st.sidebar:` | botão "Clear chat" |
| Estado da conversa | `st.session_state.messages` | lista de dicts `{role, content, sources}` |
| Bolha do chat | `st.chat_message("user" \| "assistant")` | moldura visual de cada turno |
| Markdown | `st.markdown(...)` | renderiza a resposta (e as URLs como lista) |
| Caixa de envio | `st.chat_input(...)` | retorna o texto digitado (ou `None`) |
| Loading | `st.spinner(...)` | mostra "Retrieving docs..." enquanto o backend roda |
| Sources | `st.expander("Sources")` | seção recolhida/aberta com as URLs |
| Erro | `st.error` + `st.exception` | mostra falha do backend sem quebrar a UI |
| Tema | `streamlit/config.toml` | cores dark + cor primária verde |

### Ciclo de vida de uma mensagem

```
1. Usuário digita em st.chat_input
        │
        ▼
2. Script roda de novo (rerun do Streamlit)
        │
        ├─ append da mensagem do usuário em session_state
        ├─ renderiza o histórico todo (for msg in session_state.messages)
        │
        ▼
3. Bloco com a nova pergunta
        │
        ├─ st.chat_message("user") → mostra o prompt
        └─ st.chat_message("assistant")
                │
                ├─ st.spinner + run_llm(prompt)
                ├─ st.markdown(answer)
                ├─ expander com sources (se houver)
                └─ append da resposta em session_state
        │
        ▼
4. Próximo rerun já re-renderiza o chat completo com o histórico salvo
```

### Por que `st.session_state`?

O Streamlit **não mantém variáveis Python** entre execuções do script. Sem o session state, cada pergunta apagaria o histórico da tela.

Padrão usado:

```python
if "messages" not in st.session_state:
    st.session_state.messages = [mensagem_boas_vindas]

# ao chegar resposta:
st.session_state.messages.append(
    {"role": "assistant", "content": answer, "sources": sources}
)
```

Cada item guarda **role**, **content** e **sources** juntos — assim o rerun consegue redesenhar o expander sem reprocessar o LLM.

### Clear chat

```python
if st.button("Clear chat"):
    st.session_state.pop("messages", None)
    st.rerun()
```

Remove o histórico e força um novo rerun; o `if "messages" not in ...` recria a mensagem inicial.

### Separação UI ↔ Backend

- `main.py` **não** conhece LangChain, Pinecone nem o agente — só chama `run_llm` e pinta o resultado
- `backend/core.py` **não** conhece Streamlit — devolve `{answer, context}`
- A ponte é um dicionário simples; `_format_sources` converte `Document` → lista de strings para a UI

Isso facilita testar o backend (`uv run backend/core.py`) sem subir a interface.

### Tema

`streamlit/config.toml` (precisa estar em `.streamlit/` para o Streamlit carregar automaticamente):

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#1E1E1E"
secondaryBackgroundColor = "#252526"
textColor = "#FFFFFF"
font = "sans serif"
```

---

## Decisões de implementação (para consulta futura)

### Por que a tool retorna `serialized` + `docs`?

O padrão `content_and_artifact` do LangChain separa:
- **content** → o que vai para o modelo
- **artifact** → o que a aplicação usa depois (não entra no contexto do LLM como "tool result" serializado da mesma forma)

Se só retornasse o `serialized`, você perderia a estrutura `Document` (metadata com URL). Comentário no código:

> "com o raw consigo mostrar de onde vem o documento, avaliar o RAG e debugar"

### Por que o agente é criado fora do `run_llm`?

```python
# Fora → agente criado uma única vez
agent = create_agent(model=model, tools=[retrieve_context], system_prompt=SYSTEM_PROMPT)
```

Se ficasse **dentro** de `run_llm`, toda chamada recriaria o agente (recria modelo, tools, etc.) — ineficiente. Instanciar no módulo reutiliza o mesmo agente em todas as perguntas da sessão.

### Por que deduplicar `context_docs`? (melhoria futura)

Hoje o código usa `extend` e **não** remove duplicatas. Se a tool for chamada duas vezes (ou o mesmo chunk voltar), sources podem aparecer repetidos. A deduplicação típica usa:

```
key = sha256(source + page_content)
```

Igual à lógica de IDs do `injection.py` (mesma ideia de idempotência). Assim:
- Sources não aparecem repetidos na UI
- O contexto não "pesa" duas vezes na avaliação do RAG

### Por que `k=4` no `similarity_search`?

Trade-off clássico:
- `k` menor → resposta mais focada, menos ruído, menos tokens
- `k` maior → mais cobertura, mais chance de trazer docs irrelevantes

`k=4` é um ponto de partida comum para docs técnicas. Depois de observar as respostas, você pode subir para 6–8 se faltar contexto, ou baixar se houver muito ruido.

### Por que `chunk_size=4000` e `chunk_overlap=200`?

| Parâmetro | Função | Efeito |
|---|---|---|
| `chunk_size=4000` | tamanho máximo de cada chunk (caracteres) | chunks grandes → mais contexto por busca, menos chunks no índice |
| `chunk_overlap=200` | sobreposição entre chunks vizinhos | evita cortar no meio de uma ideia e perder a continuidade |

Para documentação técnica (tutoriais longos), chunks maiores costumam funcionar melhor do que 500–1000 chars usados em PDFs densos.

### IDs determinísticos no Pinecone

```python
ids = [hashlib.sha256(f"{c.metadata['source']}::{c.page_content}".encode()).hexdigest() for c in chunks]
```

Mesmo conteúdo → mesmo ID → reexecutar o script **não duplica** vetores no índice. É idempotência de ingestão.

### `load_dotenv()` e chaves

As chaves (`OPEN_ROUTER_API_KEY`, `PINECONE_API_KEY`) vêm do `.env` via `python-dotenv`. Nunca commitar o `.env` (já está no `.gitignore`).

### LangSmith

`injection.py` usa `@traceable(name="injection")` para exportar traces. Em produção ajuda a ver custo, latência e erros do pipeline de ingestão.

---

## Pipeline de injeção (indexação)

Arquivo: `injection.py`

```
https://python.langchain.com
        │
        ▼
TavilyCrawl (max_depth=5, instructions="advanced")
        │
        ▼
Document(page_content=..., metadata={"source": url})
        │
        ▼
RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        │
        ▼
add_documents(chunks, ids=sha256(source::content))
        │
        ▼
Pinecone index: lang-docs-rag
```

### Diferença TavilyExtract vs TavilyCrawl

| Ferramenta | O que faz |
|---|---|
| **TavilyExtract** | Você entrega URLs; ele só extrai o conteúdo. Não segue links. |
| **TavilyCrawl** | Você entrega uma URL inicial; ele navega sozinho (depth/breadth) e extrai cada página. |
| **TavilyMap** | Só mapeia URLs (sem conteúdo) — útil para ver a estrutura do site. |

### Como escolher depth / breadth / limit

| Cenário | max_depth | max_breadth | limit |
|---|---|---|---|
| Docs (árvore de páginas) | 2–3 | 20–50 | 200–1000 |
| Blog / site pequeno | 1–2 | 10–20 | 50–200 |
| Site grande | baixo (1–2) | médio | alto (controla custo) |

Regra prática: comece pequeno (`depth=1`, `breadth=10`, `limit=50`), rode, veja o que volta, e aumente gradualmente.

---

## Como rodar

### 1. Requisitos

- Python `>=3.10`
- Chaves no `.env`:

```env
OPEN_ROUTER_API_KEY=...
PINECONE_API_KEY=...
```

### 2. Instalar dependências

```bash
uv sync
# ou
pip install -e .
```

### 3. Indexar a documentação (primeira vez / reindexar)

```bash
uv run injection.py
```

Sobe os chunks da docs LangChain para o Pinecone. Só precisa rodar quando a base muda (ou na primeira configuração).

### 4. Subir a UI

```bash
uv run streamlit run main.py
```

### 5. Testar o backend sem UI

```bash
uv run backend/core.py
```

Imprime o dicionário `{"answer": ..., "context": [...]}` para validar retrieval + geração.

---

## Estrutura do projeto

```
langchain_docs_rag/
├── backend/
│   └── core.py          # agente, tool retrieve_context, run_llm
├── streamlit/
│   └── config.toml      # config do tema Streamlit
├── injection.py         # crawl + chunk + upsert no Pinecone
├── main.py              # UI Streamlit (chat + sources)
├── pyproject.toml       # dependências (uv)
├── .env                 # chaves (não versionar)
└── README.md            # este documento
```

---

## Glossário rápido

| Termo | Significado |
|---|---|
| **Embedding** | representação vetorial do texto (semântica) |
| **Chunk** | pedaço de documento indexado |
| **Vector store** | banco de busca por similaridade (Pinecone) |
| **Tool / `@tool`** | função que o agente/LLM pode chamar |
| **content_and_artifact** | padrão: devolve string para o LLM + dados brutos para a app |
| **ToolMessage** | mensagem no histórico que carrega o resultado de uma tool |
| **artifact** | campo da ToolMessage com os dados originais (ex.: lista de `Document`) |
| **Agentic RAG** | RAG onde o LLM orquestra as tools de recuperação |
| **Hybrid RAG** | RAG que combina múltiplas estratégias de retrieval |
| **Idempotência** | reexecutar o mesmo passo não duplica efeito (IDs hash no Pinecone) |
