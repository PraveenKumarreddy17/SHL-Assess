# SHL Conversational Assessment Recommender

## Overview

Built a FastAPI conversational agent that recommends SHL assessments using retrieval-based matching from the SHL catalog.

The system:
- asks clarification questions for vague queries
- recommends assessments from SHL catalog only
- supports refinement in multi-turn conversations
- supports comparison queries
- refuses out-of-scope questions

## Retrieval Design

The SHL catalog was structured into JSON format.

TF-IDF retrieval was used with:
- assessment name
- description
- categories
- job levels
- languages

This enabled semantic matching between user requirements and assessments.

## Conversation Design

The API is stateless.

The full conversation history is passed into every `/chat` request.

The agent:
- asks clarifying questions when information is insufficient
- updates recommendations when constraints change
- compares assessments using catalog data
- avoids hallucinations by grounding responses in retrieved catalog entries

## Prompt Design

Retrieved catalog context is injected into the LLM prompt to ensure grounded responses.

The assistant only recommends assessments that exist in the SHL catalog.

## Evaluation

The system was tested for:
- vague queries
- recommendation generation
- conversational refinement
- comparison behavior
- out-of-scope refusal

## AI Tool Usage

ChatGPT was used for debugging, retrieval design guidance, and FastAPI improvements.