# RAG Architecture Diagram - Food Recommendation Chatbot

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG-POWERED FOOD RECOMMENDATION SYSTEM               │
└─────────────────────────────────────────────────────────────────────────┘

                              USER INPUT LAYER
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌──────▼──────┐               ┌────────▼────────┐
            │  CLI Search │               │  Chatbot Query  │
            │  Interface  │               │   Interface     │
            └──────┬──────┘               └────────┬────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Query Processing │
                          │  & Normalization  │
                          └─────────┬─────────┘
                                    │
        ╔═══════════════════════════╩══════════════════════════╗
        ║           RETRIEVAL-AUGMENTED GENERATION LAYER       ║
        ╚═════════════════════════════════════════════════════╬╝
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌──────▼──────────┐          ┌─────────▼────────┐
            │ EMBEDDING LAYER │          │ VECTOR DATABASE  │
            ├─────────────────┤          ├──────────────────┤
            │ Sentence-       │          │   ChromaDB       │
            │ Transformers    │          │                  │
            │                 │          │ • Vector Index   │
            │ (all-MiniLM-    │          │ • Food Metadata  │
            │  L6-v2)         │          │ • Embeddings     │
            │                 │          │                  │
            │ Input: Query    │          │ Data Source:     │
            │ Output:         │          │ FoodDataSet.json │
            │ 384-dim vector  │          │                  │
            └────────┬────────┘          └────────┬─────────┘
                     │                            │
                     │       ┌──────────────────┐ │
                     │       │ SEMANTIC SEARCH  │ │
                     └──────►│ (Similarity      │◄┘
                             │  Matching)       │
                             └────────┬─────────┘
                                      │
                          ┌───────────▼──────────┐
                          │ RETRIEVE CANDIDATES │
                          │ (Top-K results)     │
                          │ K=5-10 results      │
                          └───────────┬──────────┘
                                      │
        ╔═══════════════════════════════╩═════════════════════════╗
        ║        LLM AUGMENTATION & ENHANCEMENT LAYER              ║
        ╚═══════════════════════════════════════════════════════╬═╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
            ┌──────▼──────────┐            ┌───────────▼─────────┐
            │ CONTEXT BUILDER │            │  PROMPT ENGINEERING │
            ├─────────────────┤            ├─────────────────────┤
            │ • Original Query│            │ System Prompt:      │
            │ • Retrieved     │            │ "You are a food     │
            │   Items (Top-K) │            │  recommendation     │
            │ • Food Details  │            │  expert"            │
            │ • Metadata      │            │                     │
            │ • Nutrition     │            │ User Context:       │
            │   Info          │            │ Retrieved food data │
            │                 │            │                     │
            └────────┬────────┘            └─────────┬───────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  PROMPT CONSTRUCTION    │
                        ├─────────────────────────┤
                        │ Template:               │
                        │ "Based on: [context],   │
                        │  Answer: [query]"       │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │                                 │
            ┌──────▼────────────┐         ┌──────────▼───────┐
            │  GOOGLE GEMINI    │         │  API CALL        │
            │  LLM API          │         ├──────────────────┤
            ├───────────────────┤         │ • Model:         │
            │ • Model:          │         │   gemini-pro      │
            │   gemini-pro      │         │ • Temperature:   │
            │ • Temperature:    │         │   0.7             │
            │   0.7             │         │ • Max tokens:    │
            │ • Max Tokens:     │         │   500             │
            │   500             │         │ • Context:       │
            │                   │         │   Retrieved data │
            └──────┬────────────┘         └────────┬─────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                    ┌─────────────▼──────────┐
                    │  LLM RESPONSE          │
                    ├────────────────────────┤
                    │ • Personalized         │
                    │   Recommendation       │
                    │ • Explanation          │
                    │ • Nutritional Info     │
                    │ • Similar Items        │
                    └─────────────┬──────────┘
                                  │
                                  │
        ╔═══════════════════════════╩═════════════════════════╗
        ║              RESPONSE FORMATTING & OUTPUT            ║
        ╚═════════════════════════════════════════════════════╬╝
                                  │
                    ┌─────────────┴──────────┐
                    │                        │
            ┌──────▼──────┐         ┌───────▼────────┐
            │ CLI Output  │         │ Chatbot Reply  │
            ├─────────────┤         ├────────────────┤
            │ • Formatted │         │ • Natural Lang │
            │   Results   │         │ • Interactive  │
            │ • Rankings  │         │ • Follow-up    │
            │ • Details   │         │   Questions    │
            └─────────────┘         └────────────────┘
                    │                        │
                    └─────────────┬──────────┘
                                  │
                          ┌───────▼────────┐
                          │  USER DISPLAY  │
                          └────────────────┘
```

---

## Data Processing Flow

### Phase 1: Data Preparation (Offline)
```
FoodDataSet.json
        │
        ├─► Parse Food Items
        │   (name, category, nutrition, etc.)
        │
        ├─► Generate Embeddings
        │   (Sentence-Transformers)
        │
        └─► Store in ChromaDB
            (Vector Index Created)
```

### Phase 2: Query Retrieval (Online)
```
User Query: "I want a high-protein breakfast"
        │
        ├─► Embed Query (384-dim vector)
        │
        ├─► Semantic Search in ChromaDB
        │   (cosine similarity)
        │
        └─► Return Top-5 Candidates:
            1. Chicken Egg (0.92 similarity)
            2. Greek Yogurt (0.89 similarity)
            3. Oatmeal (0.87 similarity)
            4. Almonds (0.85 similarity)
            5. Cottage Cheese (0.83 similarity)
```

### Phase 3: LLM Enhancement
```
Context Assembly:
┌─────────────────────────────────┐
│ System: "You are a food expert" │
│                                 │
│ Retrieved Items:                │
│ - Chicken Egg (95g protein)    │
│ - Greek Yogurt (20g protein)   │
│ - Oatmeal (10g protein)        │
│                                 │
│ User Query:                      │
│ "I want high-protein breakfast" │
└─────────────────────────────────┘
        │
        ├─► Send to Google Gemini API
        │
        └─► Get Enhanced Response:
            "Based on your needs, I recommend:
            1. Scrambled Chicken Eggs (best protein)
            2. Greek Yogurt Bowl with Granola
            3. Overnight Oats with Almonds
            
            Why? [AI-generated explanation]
            Nutrition breakdown...
            Alternative suggestions..."
```

---

## Component Details

### Data Sources
| Component | Details |
|-----------|---------|
| **Dataset** | FoodDataSet.json |
| **Fields** | name, calories, protein, carbs, fats, fiber, category |
| **Size** | ~1000+ food items |

### Embedding Model
| Property | Value |
|----------|-------|
| **Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Output Dimension** | 384 |
| **Speed** | ~5ms per query |
| **Quality** | High semantic understanding |

### Vector Database
| Property | Value |
|----------|-------|
| **Database** | ChromaDB |
| **Storage** | In-memory + Persistent |
| **Search Algorithm** | Approximate Nearest Neighbor (ANN) |
| **Similarity Metric** | Cosine Similarity |

### LLM Integration
| Property | Value |
|----------|-------|
| **Provider** | Google Generative AI |
| **Model** | gemini-pro |
| **Temperature** | 0.7 (balanced creativity) |
| **Max Tokens** | 500 |
| **Authentication** | API Key from .env |

---

## Code Flow (Pseudocode)

```python
# 1. INITIALIZATION
embedder = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = ChromaDB()
llm = GoogleGemini(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. LOAD DATA
foods = load_json("FoodDataSet.json")
for food in foods:
    embedding = embedder.encode(food['name'])
    vector_db.add(food_id, embedding, metadata=food)

# 3. QUERY PROCESSING
user_query = "high protein breakfast"
query_embedding = embedder.encode(user_query)  # Embed

# 4. RETRIEVAL
retrieved = vector_db.search(
    query_embedding, 
    k=5  # Top 5 results
)

# 5. CONTEXT BUILDING
context = build_prompt(
    system_prompt="You are a food expert",
    retrieved_items=retrieved,
    user_query=user_query
)

# 6. LLM CALL
response = llm.generate(context)

# 7. OUTPUT
display_response(response)
```

---

## System Architecture - Component Relationships

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                        │
├──────────────────────────────────────────────────────────┤
│  • CLI Interface (search_cli.py)                         │
│  • Chatbot Interface (enhanced_rag_chatbot.py)          │
│  • User Query Input                                      │
└───────────────┬────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────┐
│                   PROCESSING LAYER                     │
├─────────────────────────────────────────────────────────┤
│  • Query Normalization & Validation                     │
│  • Embedding Generation (Sentence-Transformers)       │
│  • Semantic Search Execution                           │
│  • Result Ranking & Filtering                          │
└───────────────┬────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────┐
│                   DATA LAYER                           │
├─────────────────────────────────────────────────────────┤
│  • ChromaDB Vector Store                               │
│  • FoodDataSet.json Raw Data                           │
│  • Metadata & Nutrition Information                    │
│  • Embedding Cache                                     │
└───────────────┬────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────┐
│                   ENRICHMENT LAYER                     │
├─────────────────────────────────────────────────────────┤
│  • Google Gemini API Integration                       │
│  • Prompt Construction & Optimization                 │
│  • Context Window Management                          │
│  • Response Post-Processing                           │
└───────────────┬────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────┐
│                   OUTPUT LAYER                         │
├─────────────────────────────────────────────────────────┤
│  • Formatted Response Display                          │
│  • Conversation History Management                     │
│  • Logging & Analytics                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Key Advantages of This Architecture

✅ **Fast**: Semantic search is milliseconds, not database scans  
✅ **Smart**: Embeddings capture food similarity (e.g., breakfast items cluster)  
✅ **Accurate**: Gemini adds reasoning on top of retrieved data  
✅ **Scalable**: Can handle 1000s of food items efficiently  
✅ **Context-Aware**: LLM knows which foods were retrieved  
✅ **Hallucination Prevention**: Grounded in real food data  
✅ **Modular**: Easy to swap components (different embeddings, LLMs, databases)  

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.8+ | Backend implementation |
| Embeddings | Sentence-Transformers | Convert text to vectors |
| Vector DB | ChromaDB | Semantic search storage |
| LLM | Google Gemini API | Natural language reasoning |
| Data Source | FoodDataSet.json | Food information database |
| CLI | Python argparse | User interface for search |
| Config | python-dotenv | Environment variable management |

---

## Execution Flow Example

### User asks: "What's a healthy snack for afternoon energy?"

**Step 1:** Query Processing
- Input: "What's a healthy snack for afternoon energy?"
- Normalize & understand intent

**Step 2:** Embedding & Retrieval
- Embed query → 384-dim vector
- Search ChromaDB for similar foods
- Retrieved: Almonds, Banana, Greek Yogurt, Dark Chocolate, Mixed Nuts

**Step 3:** Context Building
```
System Prompt:
"You are an expert nutritionist and food recommendation specialist. 
Provide personalized food recommendations based on the retrieved food items."

Retrieved Context:
- Almonds: 164 cal, 6g protein, healthy fats
- Banana: 105 cal, 1.3g protein, natural sugars
- Greek Yogurt: 100 cal, 17g protein, probiotics
- Dark Chocolate: 168 cal, 12% cocoa, antioxidants
- Mixed Nuts: 200 cal, 7g protein, energy

User Query: What's a healthy snack for afternoon energy?
```

**Step 4:** LLM Generation
- Google Gemini processes context
- Generates personalized response

**Step 5:** Response
```
"For afternoon energy, I recommend Greek Yogurt with Almonds! 
Here's why: Greek yogurt provides sustained protein (17g) and 
probiotics, while almonds offer healthy fats and minerals. Together, 
they prevent the energy crash. Alternatively, a Banana with almonds 
provides quick energy from natural sugars plus long-lasting nutrition 
from fats and protein.

Nutrition: ~180 cal, 10g protein, sustained energy for 3-4 hours"
```

---

## Future Enhancement Opportunities

🔧 **Possible Improvements:**
- Multi-modal embeddings (support images)
- User preference learning
- Dietary restriction filtering
- Real-time nutrition database integration
- Caching layer for frequent queries
- Rate limiting & API optimization
- User feedback loop for continuous improvement
