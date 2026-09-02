import os
import json
from dotenv import load_dotenv
from groq import AsyncGroq
from tavily import AsyncTavilyClient

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def verify_claims(text: str) -> dict:
    """Fetches evidence from Tavily and evaluates the claim using Groq."""
    search_result = await tavily_client.search(query=text, search_depth="basic")
    context = "\n".join([result["content"] for result in search_result["results"]])

    prompt = f"""
    You are a real-time fact-checker. Determine if the following claim is TRUE, FALSE, or NUANCE based on the provided context. If the claim is not making a factual statement, assume it to be what the claim sounds most similar to
    Claim: {text}
    Context: {context}
    Respond ONLY with a valid JSON object in this exact format:
    {{"claim": "the exact claim", "status": "TRUE/FALSE/NUANCE", "reason": "1-sentence explanation"}}
    """

    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)