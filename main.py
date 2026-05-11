from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from schemas import ChatRequest, ChatResponse, Recommendation
from utils.retriever import CatalogRetriever
from utils.prompts import build_full_prompt

import os

load_dotenv()

app = FastAPI(title="SHL Conversational Assessment Recommender")
@app.get("/")
async def root():
    return {"status": "ok"}



# =========================
# LOAD RETRIEVER
# =========================
retriever = CatalogRetriever("catalog.json")


# =========================
# LOAD LLM
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================
# CATEGORY MAPPING
# =========================
CATEGORY_MAP = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C"
}


# =========================
# HEALTH ENDPOINT
# =========================
@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages required"
        )

    # =========================
    # BUILD CONVERSATION HISTORY
    # =========================
    history_text = "\n".join([
        f"{m.role}: {m.content}"
        for m in request.messages
    ])

    # =========================
    # GET LAST USER MESSAGE
    # =========================
    last_user_message = ""

    for m in reversed(request.messages):
        if m.role == "user":
            last_user_message = m.content
            break

    lower_msg = last_user_message.lower()

    # =========================
    # OUT OF SCOPE HANDLING
    # =========================
    blocked_topics = [
        "salary",
        "legal",
        "lawsuit",
        "tax",
        "politics",
        "medical",
        "religion",
        "ignore previous instructions",
        "system prompt"

    ]

    if any(word in lower_msg for word in blocked_topics):

        return ChatResponse(
            reply="I can only help with SHL assessment recommendations.",
            recommendations=[],
            end_of_conversation=False
        )

    # =========================
    # CLARIFICATION HANDLING
    # =========================
    vague_phrases = [
        "need assessment",
        "recommend assessment",
        "need a test",
        "help me hire",
        "suggest assessment"
    ]

    is_vague = (
        len(lower_msg.split()) < 5 or
        any(v in lower_msg for v in vague_phrases)
    )

    if is_vague and len(request.messages) < 4:

        return ChatResponse(
            reply=(
                "Please share:\n"
                "- role/job title\n"
                "- experience level\n"
                "- skills or traits to assess"
            ),
            recommendations=[],
            end_of_conversation=False
        )
        
        
        

        

    # =========================
    # COMPARISON HANDLING
    # =========================
    if "difference between" in lower_msg or "compare" in lower_msg:

        comparison_results = retriever.search(
            last_user_message,
            k=2
        )

        if len(comparison_results) >= 2:

            item1 = comparison_results[0]
            item2 = comparison_results[1]

            comparison_reply = f"""
{item1.get('name')}:
{item1.get('description', '')}

{item2.get('name')}:
{item2.get('description', '')}
"""

            return ChatResponse(
                reply=comparison_reply,
                recommendations=[],
                end_of_conversation=False
            )

    # =========================
    # RETRIEVE MATCHES
    # =========================
    retrieved = retriever.search(
        history_text,
        k=10
    )

    # =========================
    # BUILD CONTEXT
    # =========================
    context = "\n".join([
        f"""
Name: {item.get('name', '')}
Description: {item.get('description', '')}
Categories: {', '.join(item.get('keys', []))}
Job Levels: {', '.join(item.get('job_levels', []))}
Languages: {', '.join(item.get('languages', []))}
URL: {item.get('link', '')}
        """
        for item in retrieved
    ])

    prompt = build_full_prompt(
        history_text,
        context,
        last_user_message
    )

    # =========================
    # CALL LLM
    # =========================
    try:

        response = llm.invoke(prompt)

        assistant_reply = response.content.strip()

    except Exception as e:

        print("LLM Error:", e)

        return ChatResponse(
            reply="Sorry, there was an issue generating a response.",
            recommendations=[],
            end_of_conversation=False
        )

    # =========================
    # FORMAT RECOMMENDATIONS
    # =========================
    recommendations = []

    for item in retrieved[:5]:

        categories = item.get("keys", [])

        test_type = "General"

        for category in categories:
            if category in CATEGORY_MAP:
                test_type = CATEGORY_MAP[category]
                break

        recommendations.append(
            Recommendation(
                name=item.get("name", "Unknown"),
                url=item.get("link", ""),
                test_type=test_type
            )
        )

    # =========================
    # FINAL RESPONSE
    # =========================
    return ChatResponse(
        reply=assistant_reply,
        recommendations=recommendations,
        end_of_conversation=True
    )


# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
