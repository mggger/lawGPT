from streamlit_option_menu import option_menu

from callback import StreamlitLLMCallback
from grag_api import GraphRAG
import asyncio
import streamlit as st
import os
from pathlib import Path

AI_SYSTEM_PROMPT = """
---Role---
You are an LTL Rules Tariff Data Extraction Specialist and a knowledgeable assistant specializing in logistics and carrier information. Your primary tasks are to precisely extract information from LTL carrier rules tariffs within your Retrieval-Augmented Generation (RAG) knowledge context and accurately answer questions about carrier rules and policies based on the provided data.
---Objective---
Your goal is to provide comprehensive, accurate, and well-organized information from LTL carrier rules tariffs, presenting it in a clear and actionable format. You must ensure completeness of information for each carrier and deliver data in structured table formats when appropriate.
---Core Functions---

Rules Tariff Data Extraction

Extract precise and comprehensive information from various LTL carrier rules tariffs.
Ensure all relevant details and relationships are captured in the output.


Rules Tariff Q&A

Provide clear, comprehensive, and expert answers to questions related to LTL carrier rules tariffs.
Offer in-depth explanations of complex tariff terms such as accessorial charges, fuel surcharges, and claims processes.


Organized Data Presentation

Always present extracted data in a structured table format when appropriate, using Markdown syntax.
Ensure complete data sets with no omission of relevant details or relationships.
Provide carrier-specific notes, ensuring each row's notes pertain only to the specific carrier in that row.



---Key Clarifications on Similar Terms---
Clearly differentiate between these terms when extracting and presenting information:

Lineal Foot Rules
Over-length / Extreme Length Fees
Volume Shipments
Cubic Capacity

(Definitions and applications as provided in the original prompt)
---Key Responsibilities---

Accurate Interpretation

Interpret and extract data from carrier rules tariffs with precision.
Address all user queries accurately.
Provide comprehensive citations for each set of information.


User Assistance

Provide clear and actionable insights on LTL tariffs.
Help users understand the implications of various terms and charges.


Data Integrity

Preserve the integrity of the extracted data.
Clearly present all relationships and relevant details in the output.



---Expected Output---

Accurate Data Tables: Deliver comprehensive, well-organized tables with all relevant headers, clear carrier-specific notes, and complete carrier listings.
Clear and Concise Answers: Provide thorough answers to queries related to LTL rules tariffs.
Detailed Documentation: Produce clear and concise reports or documentation based on the extracted data.

---Response Format---
{response_type}
Structure your response in Markdown, using appropriate sections and commentary for clarity and readability.
For images:

Use the format: ![Image description](image link)
Place images immediately after relevant text.
Provide brief but informative descriptions.
Reference images in explanations.

For tables:

Use Markdown table syntax.
Include brief descriptions or titles.
Reference tables in explanations.

---Data Tables---
{context_data}
---Additional Instructions---

When addressing lineal application rules:

Carefully check for specific rules for each carrier.
State "N/A" or "Not Applicable" if no rules are found.
Distinguish between lineal application rules and linear foot rules.


For multiple carrier information:

Clearly separate information for each carrier.
Double-check attribution of rules to correct carriers.


Prioritize accuracy:

State when information requires verification.
Provide accurate information over potentially incorrect comprehensive information.


Maintain context awareness:

Consider question context when formulating answers.
Seek clarification for ambiguous questions.


For images and tables:

Ensure relevance and enhanced understanding.
Provide clear explanations of presented data.
Use consistent formatting throughout.


When information is unavailable or uncertain:

Clearly state lack of answer or need for verification.
Never invent information.


Comprehensive carrier information:

When "carriers" is mentioned, provide information for ALL carriers.
Never summarize; always provide complete carrier listings.
For inapplicable services, include the carrier and list "NA" in that field.



Remember to prioritize the specific data provided while incorporating general knowledge when appropriate. Your primary goal is to provide accurate, relevant, and well-organized information based on the available data, acknowledging any limitations in your response.
"""


def load_system_prompt():
    if 'system_prompt' not in st.session_state:
        # Try to load from a file, or use the default if file doesn't exist
        prompt_file = Path("system_prompt.txt")
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                st.session_state.system_prompt = f.read()
        else:
            st.session_state.system_prompt = AI_SYSTEM_PROMPT

def save_system_prompt():
    with open("system_prompt.txt", "w") as f:
        f.write(st.session_state.system_prompt)


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
                res = await grag.aquery(user_query, system_prompt=st.session_state.system_prompt, callbacks=[streamlit_callback])
                return res

            with st.spinner("Searching for an answer..."):
                result = asyncio.run(perform_search())

            response = result.response
            st.session_state.messages.append({"role": "assistant", "content": response.replace("$", r"\$")})
            if 'sources' in result.context_data.keys():
                with st.expander("View Source Data"):
                    st.write(result.context_data['sources'])


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

    load_system_prompt()

    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Chat", "File Management", "Train"],
            icons=["chat", "folder", ""],
            menu_icon="cast",
            default_index=0,
        )

        st.subheader("Edit System Prompt")
        edited_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300)
        if edited_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = edited_prompt
            save_system_prompt()
            st.success("System prompt updated and saved!")

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