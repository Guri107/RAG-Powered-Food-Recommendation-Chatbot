# import the necessary libraries and establish a connection to Google Gemini (Generative Language API)
from shared_functions import *
from typing import List, Dict, Any
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

# --- Gemini (Google) configuration ---
# WARNING: storing API keys in source is not recommended for production.
ENV_PATH = Path('.env')


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load environment variables from a .env file if present."""
    if not path.exists():
        return

    with path.open('r', encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_PROJECT_NAME = os.environ.get('GEMINI_PROJECT_NAME', 'projects/916304117536')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-flash-latest')
GEMINI_URL = os.environ.get(
    'GEMINI_URL',
    f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        'GEMINI_API_KEY is not configured. Copy .env.example to .env and set GEMINI_API_KEY.'
    )

# Global variables
food_items = []

# Gemini environment configuration
def call_gemini(prompt: str, timeout: int = 30) -> str:
    """Call Google Gemini generateContent endpoint and return generated text."""
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    req = urllib.request.Request(GEMINI_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_json = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = str(e)
        raise RuntimeError(f"Gemini API HTTP error: {e.code} - {body}")
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

    # Try to extract generated text from known response shapes
    text = None
    # Common: response contains 'candidates' list with 'content' parts that include 'text'
    candidates = resp_json.get("candidates") or resp_json.get("outputs") or resp_json.get("results")
    if isinstance(candidates, list) and candidates:
        cand = candidates[0]
        if isinstance(cand, dict):
            content = cand.get("content") or cand.get("contents")
            if isinstance(content, list) and content:
                parts_texts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        parts_texts.append(part["text"])
                    elif isinstance(part, str):
                        parts_texts.append(part)
                if parts_texts:
                    text = "\n".join(parts_texts)
            # fallback keys
            if not text and "output" in cand:
                out = cand.get("output")
                if isinstance(out, str):
                    text = out
            if not text and "text" in cand:
                text = cand.get("text")

    # Deep search for first string if still none
    if not text:
        def find_first_string(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    res = find_first_string(v)
                    if res:
                        return res
            if isinstance(obj, list):
                for item in obj:
                    res = find_first_string(item)
                    if res:
                        return res
            return None

        text = find_first_string(resp_json) or ""

    return text.strip()

# main function sets up the entire RAG system by loading data, creating the vector database collection, testing the LLM connection, and launching the chatbot interface
def main():
    """Main function for enhanced RAG chatbot system"""
    try:
        print(" Enhanced RAG-Powered Food Recommendation Chatbot")
        print("   Powered by Google Gemini & ChromaDB")
        print("=" * 55)
        
        # Load food data
        global food_items
        food_items = load_food_data('./FoodDataSet.json')
        print(f"Loaded {len(food_items)} food items")
        
        # Create collection for RAG system
        collection = create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {'description': 'Enhanced RAG chatbot with Google Gemini integration'}
        )
        populate_similarity_collection(collection, food_items)
        print("Vector database ready")
        
        # Test LLM connection (Gemini)
        print("🔗 Testing LLM connection with Gemini...")
        try:
            test_text = call_gemini("Hello")
            if test_text and len(test_text) > 0:
                print("LLM connection established")
            else:
                print("LLM connection failed: empty response")
                return
        except Exception as e:
            print(f"LLM connection failed: {e}")
            return
        
        # Start enhanced RAG chatbot
        enhanced_rag_food_chatbot(collection)
        
    except Exception as error:
        print(f"Error: {error}")

# Context Preparation Function - function transforms the raw search results into a structured, readable format that the LLM can understand and use effectively. It extracts key information like descriptions, nutritional data, ingredients, and health benefits, organizing them in a way that helps the LLM generate more informed responses
def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return "No relevant food items found in the database."
    
    context_parts = []
    context_parts.append("Based on your query, here are the most relevant food options from our database:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:3], 1):
        food_context = []
        food_context.append(f"Option {i}: {result['food_name']}")
        food_context.append(f"  - Description: {result['food_description']}")
        food_context.append(f"  - Cuisine: {result['cuisine_type']}")
        food_context.append(f"  - Calories: {result['food_calories_per_serving']} per serving")
        
        if result.get('food_ingredients'):
            ingredients = result['food_ingredients']
            if isinstance(ingredients, list):
                food_context.append(f"  - Key ingredients: {', '.join(ingredients[:5])}")
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")
        
        if result.get('food_health_benefits'):
            food_context.append(f"  - Health benefits: {result['food_health_benefits']}")
        
        if result.get('cooking_method'):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")
        
        if result.get('taste_profile'):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")
        
        food_context.append(f"  - Similarity score: {result['similarity_score']*100:.1f}%")
        food_context.append("")
        
        context_parts.extend(food_context)
    
    return "\n".join(context_parts)

# LLM Response Generation Function - function implements the core RAG workflow by combining information retrieval with language generation. It constructs an augmented prompt that includes both the user's original prompt and the retrieved contextual data. This prompt is then sent to IBM watsonx.ai, which provides access to the IBM Granite LLM. IBM Granite uses this combined input to generate intelligent, context-aware responses that are natural and helpful.
def generate_llm_rag_response(query: str, search_results: List[Dict]) -> str:
    """Generate response using Google Gemini with retrieved context"""
    try:
        # Prepare context from search results
        context = prepare_context_for_llm(query, search_results)
        
        # Build the prompt for the LLM
        prompt = f'''You are a helpful food recommendation assistant. A user is asking for food recommendations, and I've retrieved relevant options from a food database.

User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these recommendations match their request
4. Includes relevant details like cuisine type, calories, or health benefits
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:'''

        # Generate response using Gemini
        try:
            response_text = call_gemini(prompt)
        except Exception as e:
            print(f"LLM Error: {e}")
            return generate_fallback_response(query, search_results)

        response_text = (response_text or "").strip()
        if len(response_text) < 50:
            return generate_fallback_response(query, search_results)
        return response_text
            
    except Exception as e:
        print(f"LLM Error: {e}")
        return generate_fallback_response(query, search_results)


# Fallback Response Function - function ensures the chatbot always provides a response, even if the LLM connection fails. It uses the search results to generate a basic but helpful recommendation, maintaining system reliability.
def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""
    if not search_results:
        return "I couldn't find any food items matching your request. Try describing what you're in the mood for with different words!"
    
    top_result = search_results[0]
    response_parts = []
    
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['food_name']}.")
    response_parts.append(f"It's a {top_result['cuisine_type']} dish with {top_result['food_calories_per_serving']} calories per serving.")
    
    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great option would be {second_choice['food_name']}.")
    
    return " ".join(response_parts)

# Main Chatbot Interface
def enhanced_rag_food_chatbot(collection):
    """Enhanced RAG-powered conversational food chatbot with Google Gemini"""
    print("\n" + "="*70)
    print("ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by Google Gemini")
    print("="*70)
    print("Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("Bot: Please tell me what kind of food you're looking for!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nBot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            
            elif user_input.lower() in ['compare']:
                handle_enhanced_comparison_mode(collection)
            
            else:
                # Process the food query with enhanced RAG
                handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(user_input)
                
                # Keep conversation history manageable
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-3:]
                
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye! Hope you find something delicious!")
            break
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e}")

# Enhanced Query Handler - unction orchestrates the complete RAG process: performing vector search, generating AI responses, and displaying results. It provides transparency by showing both the AI-generated response and the underlying search results that informed it.
def handle_enhanced_rag_query(collection, query: str, conversation_history: List[str]):
    """Handle user query with enhanced RAG approach using Google Gemini"""
    print(f"\n Searching vector database for: '{query}'...")
    
    # Perform similarity search with more results for better context
    search_results = perform_similarity_search(collection, query, 3)
    
    if not search_results:
        print("Bot: I couldn't find any food items matching your request.")
        print("      Try describing what you're in the mood for with different words!")
        return
    
    print(f" Found {len(search_results)} relevant matches")
    print("🧠 Generating AI-powered response...")
    
    # Generate enhanced RAG response using Google Gemini
    ai_response = generate_llm_rag_response(query, search_results)
    
    print(f"\n Bot: {ai_response}")
    
    # Show detailed results for reference
    print(f"\n Search Results Details:")
    print("-" * 45)
    for i, result in enumerate(search_results[:3], 1):
        print(f"{i}.   {result['food_name']}")
        print(f"    {result['cuisine_type']} |  {result['food_calories_per_serving']} cal | 📈 {result['similarity_score']*100:.1f}% match")
        if i < 3:
            print()

# Enhanced Comparison Feature -  enhanced comparison mode uses the LLM to provide intelligent analysis of the differences and similarities between two food queries, offering insights that go beyond simple result matching.
def handle_enhanced_comparison_mode(collection):
    """Enhanced comparison between two food queries using LLM"""
    print("\n ENHANCED COMPARISON MODE")
    print("   Powered by AI Analysis")
    print("-" * 35)
    
    query1 = input("Enter first food query: ").strip()
    query2 = input("Enter second food query: ").strip()
    
    if not query1 or not query2:
        print(" Please enter both queries for comparison")
        return
    
    print(f"\n Analyzing '{query1}' vs '{query2}' with AI...")
    
    # Get results for both queries
    results1 = perform_similarity_search(collection, query1, 3)
    results2 = perform_similarity_search(collection, query2, 3)
    
    # Generate AI-powered comparison
    comparison_response = generate_llm_comparison(query1, query2, results1, results2)
    
    print(f"\n Bot: {comparison_response}")
    
    # Show side-by-side results
    print(f"\n DETAILED COMPARISON")
    print("=" * 60)
    print(f"{'Query 1: ' + query1[:20] + '...' if len(query1) > 20 else 'Query 1: ' + query1:<30} | {'Query 2: ' + query2[:20] + '...' if len(query2) > 20 else 'Query 2: ' + query2}")
    print("-" * 60)
    
    max_results = max(len(results1), len(results2))
    for i in range(min(max_results, 3)):
        left = f"{results1[i]['food_name']} ({results1[i]['similarity_score']*100:.0f}%)" if i < len(results1) else "---"
        right = f"{results2[i]['food_name']} ({results2[i]['similarity_score']*100:.0f}%)" if i < len(results2) else "---"
        print(f"{left[:30]:<30} | {right[:30]}")

def generate_llm_comparison(query1: str, query2: str, results1: List[Dict], results2: List[Dict]) -> str:
    """Generate AI-powered comparison between two queries"""
    try:
        context1 = prepare_context_for_llm(query1, results1[:3])
        context2 = prepare_context_for_llm(query2, results2[:3])
        
        comparison_prompt = f'''You are analyzing and comparing two different food preference queries. Please provide a thoughtful comparison.

Query 1: "{query1}"
Top Results for Query 1:
{context1}

Query 2: "{query2}"
Top Results for Query 2:
{context2}

Please provide a short comparison that:
1. Highlights the key differences between these two food preferences
2. Notes any similarities or overlaps
3. Explains which query might be better for different situations
4. Recommends the best option from each query
5. Keeps the analysis concise but insightful

Comparison:'''

        try:
            resp = call_gemini(comparison_prompt)
            if resp:
                return resp
            return generate_simple_comparison(query1, query2, results1, results2)
        except Exception:
            return generate_simple_comparison(query1, query2, results1, results2)
            
    except Exception as e:
        return generate_simple_comparison(query1, query2, results1, results2)

def generate_simple_comparison(query1: str, query2: str, results1: List[Dict], results2: List[Dict]) -> str:
    """Simple comparison fallback"""
    if not results1 and not results2:
        return "No results found for either query."
    if not results1:
        return f"Found results for '{query2}' but none for '{query1}'."
    if not results2:
        return f"Found results for '{query1}' but none for '{query2}'."
    
    return f"For '{query1}', I recommend {results1[0]['food_name']}. For '{query2}', {results2[0]['food_name']} would be perfect."

# Help Function and Entry Point
def show_enhanced_rag_help():
    """Display help information for enhanced RAG chatbot"""
    print("\n ENHANCED RAG CHATBOT HELP")
    print("=" * 45)
    print(" This chatbot uses Google Gemini to understand your")
    print("   food preferences and provide intelligent recommendations.")
    print("\nHow to get the best recommendations:")
    print("  • Be specific: 'healthy Italian pasta under 350 calories'")
    print("  • Mention preferences: 'spicy comfort food for cold weather'")
    print("  • Include context: 'light breakfast for busy morning'")
    print("  • Ask about benefits: 'protein-rich foods for workout recovery'")
    print("\nSpecial features:")
    print("  •  Vector similarity search finds relevant foods")
    print("  •  AI analysis provides contextual explanations")
    print("  •  Detailed nutritional and cuisine information")
    print("  •  Smart comparison between different preferences")
    print("\nCommands:")
    print("  • 'compare' - AI-powered comparison of two queries")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the chatbot")
    print("\nTips for better results:")
    print("  • Use natural language - talk like you would to a friend")
    print("  • Mention dietary restrictions or preferences")
    print("  • Include meal timing (breakfast, lunch, dinner)")
    print("  • Specify if you want healthy, comfort, or indulgent options")

if __name__ == "__main__":
    main()
