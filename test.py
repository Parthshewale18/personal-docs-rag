from ingest import load_pdf, split_text, create_vector_store
from retriever import create_qa_chain


def test_pipeline():

    # Step 1: Test PDF loading
    print("Testing PDF loading...")
    documents = load_pdf("Tesla_EDA_Project_Report.pdf")
    print(f"✅ Loaded {len(documents)} pages")

    # Step 2: Test text splitting
    print("\nTesting text splitting...")
    texts = split_text(documents)
    print(f"✅ Split into {len(texts)} chunks")
    print(f"   Sample chunk preview: {texts[0].page_content[:100]}...")

    # Step 3: Test vector store creation
    print("\nTesting vector store creation...")
    vector_store = create_vector_store(texts)
    print("✅ Vector store created successfully")

    # Step 4: Test retriever
    print("\nTesting retriever...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    test_query = "What is Tesla?"

    retrieved_docs = retriever.invoke(test_query)

    print(f"✅ Retrieved {len(retrieved_docs)} documents")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nDoc {i+1} preview:")
        print(doc.page_content[:150])

    # Step 5: Test full QA chain
    print("\nTesting full QA chain...")

    qa_chain = create_qa_chain(vector_store)

    response = qa_chain.invoke(test_query)

    print(f"\n✅ QA Chain response:\n{response}")


if __name__ == "__main__":
    test_pipeline()