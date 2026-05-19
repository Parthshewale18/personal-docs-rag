from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


def format_docs(docs):
    """Convert a list of Document objects into a plain string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
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