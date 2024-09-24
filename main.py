from streamlit_option_menu import option_menu

from callback import StreamlitLLMCallback
from grag_api import GraphRAG
import asyncio
import streamlit as st
import os
from pathlib import Path

AI_SYSTEM_PROMPT = """
---Role---
You are a knowledgeable assistant specializing in logistics and carrier information. Your primary task is to accurately answer questions about carrier rules and policies based on the provided data tables.

---Goal---
Generate a response that directly addresses the user's question, summarizing relevant information from the input data tables. Incorporate general knowledge when appropriate, but prioritize the specific data provided.

Key points to remember:
1. If information is not available or you're unsure, clearly state that you don't have the answer. Never invent information.
2. Distinguish between "lineal application rules" and "linear foot rules". These are different concepts and should not be confused.
3. If a carrier does not have lineal application rules, explicitly state "N/A" or "Not Applicable" for that carrier.
4. Be precise in your answers. Ensure that information provided for one carrier is not mistakenly attributed to another.
5. Double-check your responses against the provided data to minimize errors.

---Response Format---
{response_type}
Structure your response in markdown, using appropriate sections and commentary as needed for clarity and readability.

For images:
1. Use the following Markdown format:
   ![Image description](image link)
2. Place the image immediately after the text it illustrates or supports.
3. Provide a brief but informative description for each image.
4. Reference the image in your explanation, e.g., "As shown in the image below..."

For tables:
1. Use Markdown table syntax, for example:
   | Column 1 | Column 2 | Column 3 |
   |----------|----------|----------|
   | Data 1   | Data 2   | Data 3   |
2. Include a brief description or title for each table.
3. Reference the table in your explanation, e.g., "As detailed in the table below..."

---Data Tables---
{context_data}

---Additional Instructions---
1. For questions about lineal application rules:
   - Carefully check if the carrier has specific lineal application rules.
   - If no lineal application rules are found, state "N/A" or "Not Applicable" for that carrier.
   - Do not confuse lineal application rules with linear foot rules.
2. When providing information about multiple carriers:
   - Clearly separate information for each carrier.
   - Double-check that you're not accidentally attributing one carrier's rules to another.
3. Accuracy is crucial:
   - If you're not certain about a piece of information, state that it requires verification.
   - It's better to provide less information that is accurate than more information that might be incorrect.
4. Context awareness:
   - Consider the specific context of the question when formulating your answer.
   - If a question seems ambiguous, you may ask for clarification before providing an answer.
5. When including images or tables:
   - Ensure they are relevant to the question and enhance understanding.
   - Provide clear explanations of the data presented in images or tables.
   - Use consistent formatting for all images and tables throughout the response.

Remember, your primary goal is to provide accurate, relevant information based on the data provided. If the data doesn't support a comprehensive answer, acknowledge the limitations in your response. Always use Markdown formatting for images and tables to ensure clarity and consistency in your responses.
"""


grag = GraphRAG()
def load_chat_page():
    st.title("GraphRAG PDF Assistant Chatbot")
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    user_query = st.chat_input(placeholder="Ask me anything")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitLLMCallback()

            async def perform_search():
                res = await grag.aquery(user_query, system_prompt=AI_SYSTEM_PROMPT, callbacks=[streamlit_callback])
                return res

            with st.spinner("Searching for an answer..."):
                result = asyncio.run(perform_search())

            response = result.response
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_file_management_page():
    st.title("File Management")

    # Multi-file uploader
    uploaded_files = st.file_uploader("Choose PDF files to upload", type=["pdf"], accept_multiple_files=True)

    upload_dir = Path("uploads")
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0
    if uploaded_files:
        with st.spinner("Upload files..."):
            for uploaded_file in uploaded_files:
                # Save the uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                grag.upsert_pdf(file_path)
                cnt += 1
        st.success(f"Successfully uploaded {cnt} new file(s).")
    st.divider()

    st.subheader("Uploaded Files")
    uploaded_files = grag.get_all_files()
    if not uploaded_files:
        st.info("No files uploaded yet.")
    else:
        for file in uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Delete", key=f"delete_{file}"):
                    grag.delete_pdf(file)
                    st.success(f"Deleted {file}")


def train_page():
    st.title("Training Data")
    last_training_time = grag.get_last_training_time()
    if last_training_time:
        st.info(f"Last training time: {last_training_time}")
    else:
        st.info("No previous training recorded")

    st.write("Click the button below to begin training.")

    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            try:
                asyncio.run(grag.aindex())
                st.success("Training completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")


def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Chat", "File Management", "Train"],
            icons=["chat", "folder", ""],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Chat":
        load_chat_page()
    elif selected == "File Management":
        load_file_management_page()
    elif selected == "Train":
        train_page()


if __name__ == "__main__":
    st.markdown("""
            <style>
            img {
                max-height: 250px;
                width: auto;
                object-fit: contain;
            }
            </style>
        """, unsafe_allow_html=True)
    main()