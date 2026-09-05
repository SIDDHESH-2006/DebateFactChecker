import os
import json
from dotenv import load_dotenv
from groq import AsyncGroq
from tavily import AsyncTavilyClient

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def verify_claims(text: str, debate_history: list) -> dict:
    """Fetches evidence from Tavily and evaluates the claim using Groq."""
    
    # 1. Fetch live context from Tavily
    search_result = await tavily_client.search(query=text, search_depth="basic")
    context = "\n".join([result["content"] for result in search_result["results"]])

    # 2. Format the recent debate history
    history_text = "\n".join(debate_history) if debate_history else "No previous context."

    # 3. Ask Groq, providing both the history and the current claim
    prompt = f"""
    You are a real-time debate fact-checker. Your job is to evaluate the CURRENT STATEMENT. 
    Use the PREVIOUS DEBATE CONTEXT only to understand pronouns (like "he", "it", "that policy") or ongoing topics. Do not fact-check the previous context again.

    PREVIOUS DEBATE CONTEXT:
    {history_text}

    CURRENT STATEMENT:
    {text}

    EVIDENCE FROM WEB:
    {context}
    
    If the text is just conversational filler (like "Hello", "How are you"), return a status of "NUANCE" with the reason "Casual conversation."
    Respond ONLY with a valid JSON object in this exact format:
    {{"claim": "[Speaker X] said [the exact claim]", "status": "TRUE/FALSE/NUANCE", "reason": "1-sentence explanation"}}
    """

    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)