# Production-Grade Verifiable RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Chainlit**, and a cloud-hosted **Qdrant** vector database. Designed with strict citation guarantees, layout-aware document ingestion, semantic quality gating, conversational query contextualization, multi-provider model fallbacks, and structured JSON observability.

---

## 🏛 Architecture Overview

The system processes queries through a deterministic state machine orchestrated by **LangGraph**, enforcing strict grounding, semantic threshold gating, and verifiable citations.

```mermaid
flowchart TD
    %% Styling
    classDef startEnd fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc;
    classDef nodeStyle fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef decisionStyle fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#c084fc;
    classDef actionStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#a7f3d0;
    classDef rejectStyle fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fca5a5;
    classDef storageStyle fill:#172554,stroke:#38bdf8,stroke-width:2px,color:#bae6fd;

    START(["● START"]):::startEnd --> IntentClassifier["Node: intent_classifier<br/><i>(predict_intent)</i>"]:::nodeStyle

    %% Intent Routing
    IntentClassifier --> RouteIntent{"Conditional Edge:<br/>route_intent"}:::decisionStyle
    
    RouteIntent -- "greeting" --> GreetingNode["Node: greeting<br/><i>(Conversational Assistant Overview)</i>"]:::actionStyle
    GreetingNode --> END1(["● END"]):::startEnd

    RouteIntent -- "document_query /<br/>document_overview" --> RetrieveNode["Node: retrieve<br/><i>(Contextualization & Vector Search)</i>"]:::nodeStyle

    %% Retrieval Node Subgraph
    subgraph RetrievalEngine ["Retrieve Node: Contextualization & Search Workflow"]
        direction TB
        CheckHistory{"Multi-Turn Chat<br/>History Exists?"}:::decisionStyle
        CheckHistory -- "Yes (Turn >= 2)" --> ContextualizeQuery["contextualize_query<br/><i>(LLM resolves pronouns & references<br/>into standalone search query)</i>"]:::nodeStyle
        CheckHistory -- "No (Turn 1)" --> RawQuery["Use Raw User Query"]:::nodeStyle
        
        OverviewCheck{"Intent =<br/>document_overview?"}:::decisionStyle
        ContextualizeQuery --> SearchTarget
        RawQuery --> OverviewCheck
        OverviewCheck -- "Yes" --> OverviewQuery["Target: Title, Abstract,<br/>Summary metadata chunks"]:::nodeStyle
        OverviewCheck -- "No" --> SearchTarget["Unified Search Query"]:::nodeStyle
        OverviewQuery --> SearchTarget
        
        SearchTarget --> QdrantANN[("Qdrant Cloud<br/>gemini-embedding-001<br/>(k=8 chunks, 3072-dim)")]:::storageStyle
        QdrantANN --> ScoreEval["Extract Document Chunks<br/>& Cosine Similarity Scores"]:::nodeStyle
    end

    RetrieveNode --> RetrievalEngine
    RetrievalEngine --> RouteQuery{"Conditional Edge:<br/>route_query<br/>(max_score >= 0.68 OR overview?)"}:::decisionStyle

    %% Semantic Quality Gate
    RouteQuery -- "No (Score < 0.68)" --> RejectionNode["Node: rejection<br/><i>(Deterministic Rejection: 'I cannot find the answer...')</i>"]:::rejectStyle
    RejectionNode --> END2(["● END"]):::startEnd

    RouteQuery -- "Yes (Score >= 0.68)" --> GenerationNode["Node: generation<br/><i>(Structured Synthesis with Provenance)</i>"]:::actionStyle

    %% Generation Node & Fallbacks
    subgraph GenerationPipeline ["Generation Node & Resilience Engine"]
        direction TB
        ContextAssembly["Context Assembler<br/>Format [Page: X | Section: Y]<br/>Inject Last 6 Conversation Turns"]:::nodeStyle
        ContextAssembly --> PrimaryLLM["Primary Generator<br/><b>Groq (openai/gpt-oss-20b)</b><br/>LPU Fast Inference (temp=0.0)"]:::nodeStyle
        
        PrimaryLLM --> SuccessCheck{"Primary LLM<br/>Succeeded?"}:::decisionStyle
        SuccessCheck -- "Yes" --> PydanticValidation["Validate CitedAnswer Schema<br/>- is_answerable (bool)<br/>- answer (str)<br/>- source_pages (List[int])<br/>- source_sections (List[str])"]:::actionStyle
        SuccessCheck -- "No (HTTP 429/503/Error)" --> FallbackLLM["Fallback Generator<br/><b>Google Gemini (gemini-3.5-flash-lite)</b><br/>Automatic Failover"]:::actionStyle
        FallbackLLM --> PydanticValidation
    end

    GenerationNode --> GenerationPipeline
    GenerationPipeline --> AnswerableCheck{"is_answerable = True?"}:::decisionStyle
    AnswerableCheck -- "Yes" --> FinalAnswer["Synthesized Answer<br/>with 1-to-1 Page Citations"]:::actionStyle
    AnswerableCheck -- "No" --> ConservativeRejection["Default to Rejection Message<br/>Empty Citations"]:::rejectStyle

    FinalAnswer --> END3(["● END"]):::startEnd
    ConservativeRejection --> END3

    %% Persistence Sidecars
    subgraph DataPersistence ["Async Storage & Native Step Tracing"]
        SQLite[("SQLite (chat_history.db)<br/>WAL Mode • busy_timeout=60s<br/>asyncio.Lock() Serialized Writes")]:::storageStyle
        UI["Chainlit UI (app.py)<br/>cl.Step Tracing & Inline Provenance"]:::actionStyle
    end

    GreetingNode -.->|Persist Step| SQLite
    FinalAnswer -.->|Persist Step| SQLite
    ConservativeRejection -.->|Persist Step| SQLite
    RejectionNode -.->|Persist Step| SQLite
    SQLite -.->|Rehydrate Context| RetrieveNode
    FinalAnswer ==>|Stream Tokens| UI
    RejectionNode ==>|Stream Tokens| UI
    GreetingNode ==>|Stream Tokens| UI
```

---

## 📐 Key Design Decisions & Technical Architecture

### 1. Document Partitioning & Coherent Chunking Strategy (`ingestion.py`)
- **Digital Text-Only Extraction (Zero OCR Overhead)**: Assumes input documents contain digital text streams. All embedded raster images and decorative graphics are explicitly bypassed. Employs lightweight native `pypdf` extraction, executing in sub-seconds with zero OCR dependencies (`tesseract`, `poppler`, `libgl1`) and zero GPU/PyTorch downloads.
- **Stateful Section Tracking**: When iterating through layout elements, the active section is maintained in a state variable (`current_section`) whenever a header or title element is encountered. Every resulting chunk inherits this section context.
- **Strict Page-Boundary Isolation**: Unlike naive contiguous text splitters that allow chunks to span across page breaks, our buffer flushes whenever `page_number` changes. This guarantees a **strict 1-to-1 mapping** between text chunks and their physical page numbers, completely eliminating page citation hallucinations.
- **Coherent Chunk Size (1,200 chars) & Overlap (250 chars)**: Sized specifically to preserve complete contextual units (such as dataset descriptions, experimental settings, and comparative result tables) without mid-sentence truncations. Slices `pdf1_rag.pdf` into 65 clean, dense chunks.
- **Embedding Model**: `models/gemini-embedding-001` (3,072 dimensions) loaded into Qdrant Cloud with Cosine Distance.

---

### 2. Conversational Query Contextualization (`graph.py`)
In multi-turn technical dialogues, users frequently use pronouns and implicit references:
- *Turn 1:* "What was the overall detection accuracy of the proposed three-stage pipeline?"
- *Turn 2:* "How did **that** compare to using the standalone detection model by itself?"
- *Turn 3:* "Why were the **other patients** excluded?"

Vector databases cannot match pronouns against indexed document text. Before searching Qdrant:
1. When prior chat history exists, [`contextualize_query`](file:///d:/henkel_GenAI/graph.py#L380-L435) automatically invokes a lightweight LLM reformulation prompt:
   - *"How did that compare to using the standalone detection model by itself?"* $\rightarrow$ *"Detection accuracy of the standalone detection model compared to the proposed three-stage pipeline."*
   - *"Why were the other patients excluded?"* $\rightarrow$ *"Why were patients with nodules excluded from the training set?"*
2. **Hybrid Search Fusion**: The retriever queries Qdrant with both the contextualized query and original query, deduplicating candidate chunks by score.

---

### 3. Calibrated Semantic Quality Gate (Cosine Threshold: 0.68)
- Instead of unconditionally prompting the LLM when no relevant documents exist, the graph evaluates the maximum cosine similarity score from Qdrant.
- If $\max(\text{scores}) < 0.68$ (and query is not an overview request), the system routes immediately to the **Rejection Node**, which deterministically returns:
  > *"I cannot find the answer to this question in the document."*
- **Calibration Precision**:
  - Out-of-scope queries (*"What is the stock price of Apple on NASDAQ?"*) score `~0.56` $\rightarrow$ **Cleanly Rejected**.
  - Factual document inquiries score `0.70` to `0.85` $\rightarrow$ **Passed to Generation**.
- **Benefits**: Zero LLM token consumption for irrelevant questions, zero hallucination risk, and sub-100ms response time for out-of-scope queries.

---

### 4. Primary LLM with Automatic Fallback (`.with_fallbacks()`)
- **Primary Generator**: Groq (`openai/gpt-oss-20b`) provides sub-second inference on specialized LPU silicon with full tool-calling and structured Pydantic compliance.
- **Fallback Generator**: Google Gemini (`gemini-3.5-flash-lite`) is bound using LangChain's `.with_fallbacks()`. In the event of API rate limits (HTTP 429), transient service interruptions (HTTP 503), or provider downtimes, execution fails over automatically to Gemini without interrupting the user's session.

---

### 5. Pydantic Output Enforcement (`CitedAnswer`)
The generation node binds a Pydantic schema using `.with_structured_output(CitedAnswer)`:
```python
class CitedAnswer(BaseModel):
    is_answerable: bool
    answer: str
    source_pages: List[int]
    source_sections: List[str]
```
- **Guaranteed Output Schema**: The model cannot return unstructured conversational banter. It must return a validated JSON payload adhering to this schema.
- **Field Sanitization**: Built-in Pydantic validators automatically de-duplicate, sort, and cast cited pages and section strings.
- **Double-Check Grounding**: If `is_answerable` is `False`, the system deterministically clears citations and outputs the standard rejection message.

---

### 6. Production SQLite Concurrency & WAL Architecture (`database.py`)
To prevent `"database is locked"` errors under high-concurrency Chainlit streaming events:
- **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode = WAL;` and `PRAGMA synchronous = NORMAL;`. Readers never block writers, and writers never block readers.
- **Async Mutex Write Lock (`_get_write_db`)**: An internal `asyncio.Lock()` serializes all write operations (`create_step`, `update_step`, `create_element`, `update_thread`, `create_user`) cooperatively in the asyncio event loop.
- **60-Second Busy Timeout**: All connections configure `timeout=60.0` and `PRAGMA busy_timeout = 60000;`.
- **Public Universal ACL**: Implements `UniversalIdentifier` ensuring thread authorization always succeeds regardless of browser session cookies.
- **Docker Mount Auto-Detection**: Handles whether `chat_history.db` is mounted as a file or folder mount automatically.

---

### 7. Production JSON Observability (`logger.py`)
- Emits single-line JSON records to `stdout` compatible with standard cloud logging solutions (Datadog, AWS CloudWatch, GCP Cloud Logging).
- Explicitly tracks:
  - Vector similarity scores (`similarity_scores`, `max_similarity_score`)
  - Retrieval and generation latency in milliseconds (`latency_ms`)
  - Model provenance and fallback indicators (`is_fallback`, `model`)
  - Output answerability and citation volume (`citations_count`)

---

## 🔍 Native UI Step Tracing & Observability

All execution traces and routing decisions are displayed directly inside the **Chainlit UI** in real time:

### Where to Find Tracing in the UI:
1. **Interactive Step Cards in the Chat Feed**:
   - Right above each assistant response, expandable `cl.Step` cards trace every phase of the pipeline:
     - **`Intent Prediction`**: Displays the detected user intent (`greeting`, `document_overview`, or `document_query`) and chosen branch.
     - **`Retrieving documents`**: Displays candidate chunks fetched from Qdrant Cloud, individual cosine similarity scores, and the maximum score.
     - **`Evaluating similarity`**: Displays the semantic quality gating decision (verifying if $\max(\text{score}) \ge 0.68$).
     - **`Synthesizing answer with citations`**: Displays LLM execution details (context chunk count, multi-turn context turns, primary Groq / fallback Gemini provider).
   - **How to inspect**: Click directly on any step card in the chat to expand and inspect its input payloads, outputs, and execution metrics.

2. **Inline Provenance Element**:
   - At the bottom of verified answers, click the **`Source Citations & Provenance`** inline card to view strict 1-to-1 page numbers, section headers, retrieval cosine score, and context counts.

---

## 🚀 Quickstart & Deployment Instructions

Follow these exact steps to run the containerized application:

### Step 1: Clone the repository
```bash
git clone https://github.com/ZiadHegazy/Henkel_GenAI_Task.git
cd Henkel_GenAI_Task
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Ensure the following variables are configured in `.env`:
- `QDRANT_URL`: Qdrant cluster endpoint
- `QDRANT_API_KEY`: Qdrant cluster authentication token
- `GROQ_API_KEY`: Groq API key for LPU generation
- `GEMINI_API_KEY`: Google AI Studio API key (for embeddings & fallback generation)
- `GROQ_MODEL`: `openai/gpt-oss-20b` (fast primary model)
- `GEMINI_MODEL`: `gemini-3.5-flash-lite` (fallback model)
- `EMBEDDING_MODEL`: `models/gemini-embedding-001` (3072-dim active Google AI Studio model)
- `SIMILARITY_THRESHOLD`: `0.68`
- `TOP_K_CHUNKS`: `8`
- `CHAINLIT_AUTH_SECRET`: Random 32+ character hex string for signing session JWTs
- `CHAT_DB_PATH`: Path to SQLite database file (`chat_history.db`)

### Step 3: Build the Docker Image
```bash
docker build -t henkel_rag_task .
```

### Step 4: Run the Application Container
```bash
docker run -p 8000:8000 --env-file .env -v ${PWD}/chat_history.db:/app/chat_history.db henkel_rag_task
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**  
Access is completely public (no login required). Your conversation threads and step telemetry automatically persist to `chat_history.db`.

---

## 📄 Document Ingestion (Local or Containerized)

The ingestion script extracts digital text, preserves 1-to-1 page boundaries, updates section headers, and populates the Qdrant vector index.

```bash
# Ingest a PDF document into Qdrant Cloud (Local or inside Docker)
python ingestion.py --file pdf1_rag.pdf --collection rag_documents

# Ingest and recreate collection from scratch
python ingestion.py --file pdf1_rag.pdf --collection rag_documents --recreate
```

> [!TIP]
> **Zero OCR Overhead**: Ingestion operates in fast digital text mode, bypassing all embedded images. The Docker image does not require `tesseract-ocr`, `poppler-utils`, `libgl1`, or heavy PyTorch / CUDA models, resulting in an ultra-lightweight image that builds in seconds.
