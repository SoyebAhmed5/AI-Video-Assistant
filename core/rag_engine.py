import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_generation_chain():
    """Create the prompt -> LLM -> output parser chain."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the transcript context provided.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return prompt | get_llm() | StrOutputParser()


def build_rag_chain(transcript: str):
    """Build a RAG system using the provided transcript."""

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    generation_chain = build_generation_chain()

    return {
        "retriever": retriever,
        "generation_chain": generation_chain,
    }


def load_rag_chain():
    """Load an existing vector store and build the RAG system."""

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    generation_chain = build_generation_chain()

    return {
        "retriever": retriever,
        "generation_chain": generation_chain,
    }


def retrieval_fallback(docs):
    """Return useful transcript content without requiring an LLM."""

    if not docs:
        return "No relevant information was found in the video transcript."

    response = (
        "⚠️ AI generation is currently unavailable.\n"
        "Here is the relevant information retrieved from your video:\n\n"
    )

    for i, doc in enumerate(docs, start=1):
        response += f"--- Relevant passage {i} ---\n"
        response += doc.page_content + "\n\n"

    return response


def ask_question(rag_system, question: str) -> str:
    print(f"\nQuestion: {question}")

    retriever = rag_system["retriever"]
    generation_chain = rag_system["generation_chain"]

    # Step 1: Retrieve relevant transcript chunks
    try:
        docs = retriever.invoke(question)

        print(f"Retrieved {len(docs)} relevant chunk(s).")

        if not docs:
            answer = "I could not find relevant information in the video transcript."
            print(f"Answer: {answer}")
            return answer

    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")

        return "Sorry, I couldn't retrieve information from the video transcript."

    # Step 2: Generate an AI answer using retrieved context
    try:
        context = format_docs(docs)

        answer = generation_chain.invoke({
            "context": context,
            "question": question,
        })

        print(f"Answer: {answer}")
        return answer

    except Exception as e:
        print(f"⚠️ AI generation failed: {e}")

        # Step 3: If the LLM fails, return retrieved content
        answer = retrieval_fallback(docs)

        print(f"Fallback answer:\n{answer}")
        return answer