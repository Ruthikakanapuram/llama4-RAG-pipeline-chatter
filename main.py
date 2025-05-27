# ========== Standard Library ==========
import os
import tempfile
import zipfile
from typing import List, Optional, Tuple, Union
import collections

from dotenv import load_dotenv

# ========== Third-Party Libraries ==========
import gradio as gr
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import Docx2txtLoader


# ========== Configs ==========
TITLE = """<h1 align="center">🗨️🦙 Llama 4 Docx Chatter</h1>"""
AVATAR_IMAGES = (
    None,
    "./logo.png",
)

# Acceptable file extensions
TEXT_EXTENSIONS = [".docx", ".zip"]
load_dotenv()
# ========== Models & Clients ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
# llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=GROQ_API_KEY)
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# ========== Core Components ==========
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=3000,
    # chunk_overlap=100,
    # separators=["\n\n", "\n"],
    # chunk_size=1000,
    chunk_size=1000,
    chunk_overlap=300,
    # separators=["\n\n", "\n", ".", " "], 
    separators=["\n\n", "\n", ".", " "]
)

# rag_template = """You are an expert assistant tasked with answering questions based on the provided documents.
# Use only the given context to generate your answer.
# If the answer cannot be found in the context, clearly state that you do not know.
# Be detailed and precise in your response, but avoid mentioning or referencing the context itself.
# rag_template = """You are an expert software testing engineer with extensive experience in analyzing Software Requirement Specifications (SRS).

# You always read the SRS document thoroughly from beginning to end to deeply understand the functional requirements and their dependencies. Based on your detailed understanding, you write comprehensive test scenarios for each requirement and comprehensive set of test cases in IEEE 829 format for each test scenario that ensure 100% test coverage of all functional requirements.
# rag_template = """You are an expert software test engineer and quality analyst with deep knowledge of IEEE 829 test case standards. You specialize in analyzing Software Requirements Specifications (SRS) and generating comprehensive test scenarios and test cases. For any given document, your goal is to:

# Thoroughly understand all functional and non-functional requirements.
# Extract and categorize test scenarios.
# For each scenario, create detailed test cases that cover base conditions, edge cases, positive flows, and negative paths.
# Present the output strictly in IEEE 829 test case specification format.

# Your output should be complete, unambiguous, and useful for immediate use in software testing. Do not leave out any requirement mentioned in the SRS. Be exhaustive, yet clear.
rag_template = """You are an expert Software Test Engineer and Quality Analyst with deep knowledge of IEEE 829 Test Case Specification standards. You specialize in analyzing Software Requirements Specification (SRS) documents and converting each requirement into a comprehensive suite of test scenarios and test cases.

For every document you analyze, your goals are:

🔹 Requirement-to-Test Scenario Mapping
Treat each individual requirement in the SRS—whether functional or non-functional—as the root of a distinct test scenario group.

Do not group multiple requirements into a single scenario. Even if multiple requirements belong to the same module or functionality, create separate test scenarios per requirement to ensure 100% traceability and coverage.

Consider explicit, implicit, dependent, cross-referenced, and repeated requirements. Every unique testcase must lead to a corresponding test scenario.

🔹 Test Scenario Generation
For every individual requirement:

Generate as many appropriate high-level test scenarios as possible that reflects the exact intent of every requirement.
Categorize each scenario module-wise or feature-wise for traceability, but do not group multiple requirements into one.

Include:

Functional test scenarios

Non-functional test scenarios such as:

Validation

Usability

Performance

Security

Compatibility

🔹 Test Case Generation
For each test scenario, create a comprehensive set of test cases that:

Guarantee 100% coverage of the requirement

Include the following:

Base cases (normal flows)

Edge cases (boundary and limit conditions)

Positive test conditions (valid paths)

Negative test conditions (invalid inputs, exceptions, and error handling)

Validation test cases (e.g., data correctness, input validation)

Usability test cases (e.g., UI accessibility, user experience expectations)

🔹 Output Format (IEEE 829 Standard)
Present each test case in this exact format:

Test Case ID

Test Case Description

Pre-Conditions

Test Steps

Test Data

Expected Result

Actual Result (Leave this blank)

Status (Leave this blank)

Remarks (Optional)

🔹 Output Expectations
Be complete and exhaustive — cover every requirement individually.

Include all test scenarios and test cases required to fully validate the system, including non-obvious cases arising from dependencies or overlaps.

Do not group or merge separate requirements into one scenario.

Be clear, structured, and professional. Your output should be ready for use in a software testing cycle without requiring post-processing.
Context:
{context}
Question:
{question}
Answer:"""
rag_prompt = PromptTemplate.from_template(rag_template)


# ========== App State ==========
class AppState:
    vectorstore: Optional[InMemoryVectorStore] = None
    rag_chain = None


state = AppState()

# ========== Utility Functions ==========


def load_documents_from_files(files: List[str]) -> List:
    """Load documents from uploaded files directly without moving."""
    all_documents = []

    # Temporary directory if ZIP needs extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".zip":
                # Extract ZIP inside temp_dir
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Load all docx from extracted zip
                loader = DirectoryLoader(
                    path=temp_dir,
                    glob="**/*.docx",
                    use_multithreading=True,
                )
                docs = loader.load()
                all_documents.extend(docs)

            elif ext == ".docx":
                # Load single docx directly
                # loader = UnstructuredFileLoader(file_path)
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                all_documents.extend(docs)

    return all_documents


def get_last_user_message(chatbot: List[Union[gr.ChatMessage, dict]]) -> Optional[str]:
    """Get last user prompt."""
    for message in reversed(chatbot):
        content = (
            message.get("content") if isinstance(message, dict) else message.content
        )
        if (
            message.get("role") if isinstance(message, dict) else message.role
        ) == "user":
            return content
    return None


# ========== Main Logic ==========




def upload_files(
    files: Optional[List[str]], chatbot: List[Union[gr.ChatMessage, dict]]
):
    """Handle file upload - .docx or .zip containing docx."""
    if not files:
        return chatbot

    file_summaries = []  # <-- Collect formatted file/folder info
    documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in files:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".zip":
                file_summaries.append(f"📦 **{filename}** (ZIP file) contains:")
                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                        zip_contents = zip_ref.namelist()

                        # Group files by folder
                        folder_map = collections.defaultdict(list)
                        for item in zip_contents:
                            if item.endswith("/"):
                                continue  # skip folder entries themselves
                            folder = os.path.dirname(item)
                            file_name = os.path.basename(item)
                            folder_map[folder].append(file_name)

                        # Format nicely
                        for folder, files_in_folder in folder_map.items():
                            if folder:
                                file_summaries.append(f"📂 {folder}/")
                            else:
                                file_summaries.append(f"📄 (root)")
                            for f in files_in_folder:
                                file_summaries.append(f"   - {f}")

                    # Load docx files extracted from ZIP
                    loader = DirectoryLoader(
                        path=temp_dir,
                        glob="**/*.docx",
                        # loader_cls=Docx2txtLoader,
                        use_multithreading=True,
                    )
                    docs = loader.load()
                    documents.extend(docs)

                except zipfile.BadZipFile:
                    chatbot.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=f"❌ Failed to open ZIP file: {filename}",
                        )
                    )

            elif ext == ".docx":
                file_summaries.append(f"📄 **{filename}**")
                # loader = UnstructuredFileLoader(file_path)
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            else:
                file_summaries.append(f"❌ Unsupported file type: {filename}")

    if not documents:
        chatbot.append(
            gr.ChatMessage(
                role="assistant", content="No valid .docx files found in upload."
            )
        )
        return chatbot

    # Split documents
    chunks = text_splitter.split_documents(documents)
    print("chunks",chunks)
    print("chunks",len(chunks))
    if not chunks:
        chatbot.append(
            gr.ChatMessage(
                role="assistant", content="Failed to split documents into chunks."
            )
        )
        return chatbot

    # Create Vectorstore
    state.vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embed_model,
    )
    retriever = state.vectorstore.as_retriever()

    # Build RAG Chain
    state.rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Final display
    chatbot.append(
        gr.ChatMessage(
            role="assistant",
            content="**Uploaded Files:**\n"
            + "\n".join(file_summaries)
            + "\n\n✅ Ready to chat!",
        )
    )
    return chatbot


def user_message(
    text_prompt: str, chatbot: List[Union[gr.ChatMessage, dict]]
) -> Tuple[str, List[Union[gr.ChatMessage, dict]]]:
    """Add user's text input to conversation."""
    if text_prompt.strip():
        chatbot.append(gr.ChatMessage(role="user", content=text_prompt))
    return "", chatbot


def process_query(
    chatbot: List[Union[gr.ChatMessage, dict]],
) -> List[Union[gr.ChatMessage, dict]]:
    """Process user's query through RAG pipeline."""
    prompt = get_last_user_message(chatbot)
    if not prompt:
        chatbot.append(
            gr.ChatMessage(role="assistant", content="Please type a question first.")
        )
        return chatbot

    if state.rag_chain is None:
        chatbot.append(
            gr.ChatMessage(role="assistant", content="Please upload documents first.")
        )
        return chatbot

    chatbot.append(gr.ChatMessage(role="assistant", content="Thinking..."))

    try:
        response = state.rag_chain.invoke(prompt)
        print("Response",response)
        chatbot[-1].content = response
    except Exception as e:
        chatbot[-1].content = f"Error: {str(e)}"

    return chatbot

def reset_app(
    chatbot: List[Union[gr.ChatMessage, dict]],
) -> List[Union[gr.ChatMessage, dict]]:
    """Reset application state."""
    state.vectorstore = None
    state.rag_chain = None
    return [
        gr.ChatMessage(
            role="assistant", content="App reset! Upload new documents to start."
        )
    ]


# ========== UI Layout ==========

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(TITLE)
    chatbot = gr.Chatbot(
        label="Llama 4 RAG",
        type="messages",
        bubble_full_width=False,
        avatar_images=AVATAR_IMAGES,
        scale=2,
        height=350,
    )

    with gr.Row(equal_height=True):
        text_prompt = gr.Textbox(
            placeholder="Ask a question...", show_label=False, autofocus=True, scale=28
        )
        send_button = gr.Button(
            value="Send",
            variant="primary",
            scale=1,
            min_width=80,
        )
        upload_button = gr.UploadButton(
            label="Upload",
            file_count="multiple",
            file_types=TEXT_EXTENSIONS,
            scale=1,
            min_width=80,
        )
        reset_button = gr.Button(
            value="Reset",
            variant="stop",
            scale=1,
            min_width=80,
        )

    send_button.click(
        fn=user_message,
        inputs=[text_prompt, chatbot],
        outputs=[text_prompt, chatbot],
        queue=False,
    ).then(fn=process_query, inputs=[chatbot], outputs=[chatbot])

    text_prompt.submit(
        fn=user_message,
        inputs=[text_prompt, chatbot],
        outputs=[text_prompt, chatbot],
        queue=False,
    ).then(fn=process_query, inputs=[chatbot], outputs=[chatbot])

    upload_button.upload(
        fn=upload_files, inputs=[upload_button, chatbot], outputs=[chatbot], queue=False
    )
    reset_button.click(fn=reset_app, inputs=[chatbot], outputs=[chatbot], queue=False)

demo.queue().launch()