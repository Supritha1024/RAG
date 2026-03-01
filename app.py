import streamlit as st
import tempfile
from utils import process_pdf, create_vectorstore, create_qa_chain

st.set_page_config(page_title="RAG Document QA", layout="wide")

st.title("📄 Document Question Answering System (RAG)")
st.write("Upload a PDF document and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    with st.spinner("Processing document..."):
        chunks = process_pdf(file_path)
        vectorstore = create_vectorstore(chunks)
        qa_chain = create_qa_chain(vectorstore)

    st.success("Document processed successfully!")

    question = st.text_input("Ask a question from the document:")

    if question:
        with st.spinner("Generating answer..."):
            response = qa_chain(question)

        st.subheader("📌 Final Answer")
        st.write(response["result"])

        st.subheader("📚 Source Chunks")
        for i, doc in enumerate(response["source_documents"]):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write("---")