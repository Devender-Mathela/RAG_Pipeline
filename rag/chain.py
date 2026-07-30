from retrieval.hybrid_search import hybrid_search
from rag.prompt import RAG_PROMPT
from llm.model import load_llm

llm=load_llm()

def rag_chain(question:str):
    """
    Execute the complete RAG pipeline
    """

    #Step 1 : Retrieve the relevant chunks
    retrieved_chunks=hybrid_search(
        query=question,
        k=5
    )

    #Step 2: Convert documents into context
    context="\n\n".join(doc["text"] for doc in retrieved_chunks)

    #Step 3: Create prompt
    prompt=RAG_PROMPT.invoke(
        {
            "context":context,
            "question":question
        }
    )

    #Step 4: Generate resposne
    response=llm.invoke(prompt)

    return {
        "answer":response.content,
        "sources":retrieved_chunks
    }

if __name__ == "__main__":

    question = " What is self attention? "

    answer = rag_chain(question)

    print("\nAnswer:\n")
    print(answer)