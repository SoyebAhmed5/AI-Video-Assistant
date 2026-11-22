from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    
    return splitter.split_text(transcript)

# def summarize(transcript: str) -> str:
    llm = get_llm()
    
    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = []

    # Step 1: Summarize each transcript chunk
    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")

        try:
            summary = map_chain.invoke({"text": chunk})
            chunk_summaries.append(summary)

        except Exception as e:
            print(f"⚠️ Chunk {i} summarization failed: {e}")

            # Preserve the original chunk if AI summarization fails
            chunk_summaries.append(chunk)

    # Step 2: Combine chunk summaries
    combined = "\n\n".join(chunk_summaries)

    # If all chunk calls failed, don't waste another API call
    if all(summary == chunk for summary, chunk in zip(chunk_summaries, chunks)):
        print("⚠️ AI summarization unavailable. Returning original transcript.")
        return transcript

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    # Step 3: Generate the final summary
    try:
        return combined_chain.invoke({"text": combined})

    except Exception as e:
        print(f"⚠️ Final summary generation failed: {e}")

        # Return collected summaries instead of crashing
        return combined

def generate_title(transcript : str) -> str:
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    try:
        result = title_chain.invoke(transcript[:2000])
        return result.content.strip()

    except Exception as e:
        print(f"Title generation failed: {e}")

        # Fallback title when AI is unavailable
        words = transcript.strip().split()
        fallback_title = " ".join(words[:8])

        if fallback_title:
            return fallback_title.capitalize() + "..."

        return "Untitled Video"