import streamlit as st
from transformers import pipeline
from models.model import load_model

# Load the fine-tuned model
model, tokenizer = load_model()
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("Question Answering Bot")

context = st.text_area("Enter the context (relevant text):")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.write(f"Answer: {result['answer']}")
