from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv(override=True)


def format_docs(docs):
    """Convert a list of Document objects into a plain string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context provided.

    Context: {context}

    Question: {question}
    """)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain