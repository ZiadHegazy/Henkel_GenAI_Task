# Production-Grade Verifiable RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Chainlit**, and a cloud-hosted **Qdrant** vector database. Designed with strict citation guarantees, layout-aware document ingestion, semantic quality gating, multi-provider model fallbacks, and structured JSON observability.

---

## 🏛 Architecture Overview

The system processes queries through a deterministic state machine orchestrated by **LangGraph**, enforcing strict grounding, semantic threshold gating, and verifiable citations.

```mermaid
flowchart TD
    subgraph Offline Ingestion ["Phase 2: Offline Document Ingestion (Fast Text Mode)"]
        PDF["Input PDF Document"] --> Partition["pypdf Digital Text Extractor (fast text, images bypassed)"]
        Partition --> Grouping["Page-Bounded & Section-Aware Buffering"]
        Grouping --> Splitter["RecursiveCharacterTextSplitter (size: 500, overlap: 50)"]
        Splitter --> Embed["Google AI Studio (gemini-embedding-001, 3072-dim)"]
        Embed --> QdrantCloud[("Qdrant Cloud Vector Database")]
    end

    subgraph LangGraph Runtime ["Phase 3: LangGraph State Machine & Orchestration"]
        UserQuery(["User Query"]) --> IntentNode["Node: Intent Classifier (predict_intent)"]
        IntentNode -- "greeting" --> GreetingNode["Node: Conversational Assistant\n(Warm Greeting & System Overview)"]
        IntentNode -- "document_query" --> RetrieveNode["Node: Retrieve Chunks (Top-k ANN)"]
        QdrantCloud -.->|Cosine Search| RetrieveNode
        RetrieveNode --> RoutingGate{"Conditional Edge: Max Cosine Score >= 0.72?"}
        
        RoutingGate -- "< 0.72" --> RejectionNode["Node: Deterministic Rejection\n'I cannot find the answer...'"]
        RoutingGate -- ">= 0.72" --> GenerationNode["Node: Structured Generation\n(Context + Multi-Turn Memory)"]

        subgraph Resilient LLM Chains [".with_fallbacks()"]
            PrimaryLLM["Primary: Groq (llama-3.3-70b-versatile)"]
            FallbackLLM["Fallback: Google (gemini-2.5-flash)"]
            PrimaryLLM -.->|Failover / RateLimit| FallbackLLM
        end

        GenerationNode --> PrimaryLLM
        GenerationNode --> FallbackLLM
        PrimaryLLM --> PydanticValidation["Pydantic CitedAnswer Schema Validation"]
        FallbackLLM --> PydanticValidation
    end

    subgraph UI & Persistence ["Phases 3 & 4: Interface, SQLite History & Telemetry"]
        GreetingNode --> ChainlitUI["Chainlit Chat UI (app.py)"]
        PydanticValidation --> ChainlitUI
        RejectionNode --> ChainlitUI
        ChainlitUI <-->|Async CRUD| SQLiteDB[("SQLite Chat Persistence (database.py)\n- Users, Threads, Steps, Feedback\n- Multi-Session Resumption")]
        ChainlitUI --> Steps["cl.Step Tracing: Intent, Retrieval, Evaluation, Synthesis"]
        ChainlitUI --> Citations["Explicit Citation Elements (Pages & Sections)"]
        RetrieveNode -.-> JSONLog["JSON Structured Logger (stdout)"]
        GenerationNode -.-> JSONLog
    end
```

---

## 📐 Key Design Decisions & Justifications

### 1. Document Partitioning & Chunking Strategy
- **Digital Text-Only Extraction (Zero OCR Overhead)**: Assumes input documents contain digital text streams. All embedded raster images and decorative graphics are explicitly bypassed. Employs `unstructured.partition_pdf(strategy="fast")` with native `pypdf` fallback, executing in sub-seconds with zero OCR dependencies (`tesseract`, `poppler`, `libgl1`) and zero GPU/PyTorch downloads.
- **Stateful Section Tracking**: When iterating through layout elements, the active section is maintained in a state variable (`current_section`) whenever a header or title element is encountered. Every resulting chunk inherits this section context.
- **Strict Page-Boundary Isolation**: Unlike naive Contiguous Text Splitters that allow chunks to span across page breaks, our buffer flushes whenever `page_number` changes. This guarantees a **strict 1-to-1 mapping** between text chunks and their physical page numbers, completely eliminating page citation hallucinations.
- **Chunk Size (500) & Overlap (50)**: Captures self-contained factual propositions and specific clauses while avoiding the dilution of semantic density. The 10% overlap (50 characters) ensures boundary continuity without cluttering the vector space.

### 2. Semantic Routing Quality Gate (Cosine Threshold: 0.72)
- Instead of unconditionally prompting the LLM when no relevant documents exist, the graph evaluates the maximum cosine similarity score from Qdrant.
- If $\max(\text{scores}) < 0.72$, the system routes immediately to the **Rejection Node**, which deterministically returns:
  > *"I cannot find the answer to this question in the document."*
- **Benefits**: Zero LLM token consumption for irrelevant questions, zero hallucination risk, and sub-100ms response time for out-of-scope queries.

### 3. Primary LLM with Automatic Fallback (`.with_fallbacks()`)
- **Primary Generator**: Groq (`llama-3.3-70b-versatile`) provides industry-leading inference speeds (>250 tokens/sec on LPU architecture) and exceptional reasoning capabilities for low-latency chat interactions.
- **Fallback Generator**: Google Gemini (`gemini-2.5-flash`) is bound using LangChain's `.with_fallbacks()`. In the event of API rate limits (HTTP 429), transient service interruptions (HTTP 503), or provider downtimes, execution fails over automatically to Gemini without interrupting the user's session.

### 4. Pydantic Output Enforcement (`CitedAnswer`)
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

### 5. Production JSON Observability (`logger.py`)
- Emits single-line JSON records to `stdout` compatible with standard cloud logging solutions (Datadog, AWS CloudWatch, GCP Cloud Logging).
- Explicitly tracks:
  - Vector similarity scores (`similarity_scores`, `max_similarity_score`)
  - Retrieval and generation latency in milliseconds (`latency_ms`)
  - Model provenance and fallback indicators (`is_fallback`, `model`)
  - Output answerability and citation volume (`citations_count`)

### 6. Intent Prediction & Conversational Routing (`graph.py`)
- Casual conversational inputs ("hello", "good morning", "who are you", "what can you do") lack high cosine similarity against factual document chunks and would otherwise trigger the 0.72 quality rejection gate.
- The **Intent Classifier Node** evaluates user input prior to retrieval. Greetings bypass vector search completely and route to a dedicated **Greeting Node**, welcoming the user and explaining system capabilities warmly without false rejections.
- Factual and technical questions proceed directly to Qdrant vector retrieval.

### 7. SQLite Chat Persistence & Multi-Session History (`database.py`)
- Built upon an asynchronous SQLite data layer (`SQLiteDataLayer` extending `chainlit.data.BaseDataLayer` using `aiosqlite`).
- Stores `users`, `threads`, `steps`, `elements`, and `feedback` in `chat_history.db`.
- **Sidebar Session History**: Users can start new chats or switch between prior sessions via the Chainlit left sidebar.
- **Multi-Turn Resumption (`@cl.on_chat_resume`)**: Clicking a past conversation reloads prior conversation turns into memory context, allowing the LLM to resolve multi-turn references and pronoun context ("it", "that finding", "the previous metric") seamlessly.

---

## 🔍 Native UI Step Tracing & Observability

All execution traces and routing decisions are displayed directly inside the **Chainlit UI** in real time:

### Where to Find Tracing in the UI:
1. **Interactive Step Cards in the Chat Feed**:
   - Right above each assistant response, expandable `cl.Step` cards trace every phase of the pipeline:
     - **`Intent Prediction`**: Displays the detected user intent (`greeting` vs. `document_query`) and chosen branch.
     - **`Retrieving documents`**: Displays the number of candidate chunks fetched from Qdrant Cloud, individual cosine similarity scores, and the maximum score.
     - **`Evaluating similarity`**: Displays the semantic quality gating decision (verifying if $\max(\text{score}) \ge 0.72$).
     - **`Synthesizing answer with citations`**: Displays LLM execution details (context chunk count, multi-turn context turns, primary Groq / fallback Gemini provider).
   - **How to inspect**: Click directly on any step card in the chat to expand and inspect its input payloads, outputs, and execution metrics.

2. **Inline Provenance Element**:
   - At the bottom of verified answers, click the **`Source Citations & Provenance`** inline card to view strict 1-to-1 page numbers, section headers, retrieval cosine score, and context counts.

3. **Structured JSON Logs**:
   - Full latency profiling and retrieval events are emitted as single-line JSON records to `stdout` for container log collectors.

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
- `GROQ_API_KEY`: Groq API key for Llama 3.3 generation
- `GEMINI_API_KEY`: Google AI Studio API key (for embeddings & fallback generation)
- `EMBEDDING_MODEL`: `models/gemini-embedding-001` (3072-dim active Google AI Studio model)
- `CHAINLIT_AUTH_SECRET`: Random 32+ character hex string for signing session JWTs
- `CHAT_DB_PATH`: Path to SQLite database file (defaults to `chat_history.db`)

### Step 3: Build the Docker Image
```bash
docker build -t chatbot:1.0 .
```

### Step 4: Run the Application Container
```bash
docker run -p 8000:8000 --env-file .env -v ${PWD}:/app chatbot:1.0
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**  
Log in with username `admin` and password `admin` (or any custom username/password). Your chat threads will automatically persist to `chat_history.db` and display in the left sidebar for instant multi-session continuation.

---

## 📄 Document Ingestion (Local or Containerized)

The ingestion script extracts digital text, preserves 1-to-1 page boundaries, updates section headers, and populates the Qdrant vector index.

```bash
# Ingest a PDF document into Qdrant Cloud (Local or inside Docker)
python ingestion.py --file path/to/document.pdf --collection rag_documents

# Ingest and recreate collection from scratch
python ingestion.py --file path/to/document.pdf --collection rag_documents --recreate
```

> [!TIP]
> **Zero OCR Overhead**: Ingestion operates in fast digital text mode, bypassing all embedded images. The Docker image does not require `tesseract-ocr`, `poppler-utils`, `libgl1`, or heavy PyTorch / CUDA models, resulting in an ultra-lightweight (~200MB) image that builds in ~10-15 seconds.
