import os
import time
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document

# Import Cleanlab evaluation tools if API key provided
try:
    from cleanlab_codex.validator import Validator
    from cleanlab_tlm.utils.rag import Eval, get_default_evals
    has_cleanlab = True
except ImportError:
    has_cleanlab = False

# Load environment variables
def load_env_variables():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        st.warning("python-dotenv not installed. Environment variables must be set manually.")

# Set environment variables if not already set
def set_api_keys():
    """Set API keys from environment variables or Streamlit secrets"""
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", "")
    if 'CLEANLAB_TLM_API_KEY' not in st.session_state:
        st.session_state.CLEANLAB_TLM_API_KEY = os.getenv('CLEANLAB_TLM_API_KEY', '')
    if 'CODEX_API_KEY' not in st.session_state:
        st.session_state.CODEX_API_KEY = os.getenv('CODEX_API_KEY', '')
    if 'CODEX_PROJECT_KEY' not in st.session_state:
        st.session_state.CODEX_PROJECT_KEY = os.getenv('CODEX_PROJECT_KEY', '')

# Initialize Streamlit state
def init_session_state():
    """Initialize session state variables"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []
    if 'response' not in st.session_state:
        st.session_state.response = ""
    if 'remediation_instructions' not in st.session_state:
        st.session_state.remediation_instructions = ""
    if 'remediated_response' not in st.session_state:
        st.session_state.remediated_response = None
    if 'remediated_eval_results' not in st.session_state:
        st.session_state.remediated_eval_results = None
    if 'saved_remediations' not in st.session_state:
        st.session_state.saved_remediations = {}

# Set page configuration
def set_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Document Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Main function to run the app
def main():
    # Load environment variables
    load_env_variables()
    
    # Initialize session state
    init_session_state()
    
    # Set API keys
    set_api_keys()
    
    # Set page configuration
    set_page_config()
    
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .document-preview {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }
        .highlight {
            background-color: #ffff00;
            padding: 0 2px;
        }
        .small-text {
            font-size: 0.8rem;
        }
        .st-emotion-cache-16idsys p {
            word-break: break-word;
        }
        .chunk-header {
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }
        .temporal-warning {
            color: #d9534f;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []
    if 'response' not in st.session_state:
        st.session_state.response = ""
    if 'remediation_instructions' not in st.session_state:
        st.session_state.remediation_instructions = ""
    if 'remediated_response' not in st.session_state:
        st.session_state.remediated_response = None
    if 'remediated_eval_results' not in st.session_state:
        st.session_state.remediated_eval_results = None
    if 'saved_remediations' not in st.session_state:
        st.session_state.saved_remediations = {}

    def load_pdf_file(uploaded_file):
        """Load a single uploaded PDF file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Add file name to metadata for better tracking
            for doc in documents:
                doc.metadata['source_name'] = uploaded_file.name
                
            return documents
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return []
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
        """Split documents into manageable chunks"""
        if not documents:
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        return chunked_documents

    def create_vector_store(documents, openai_api_key):
        """Create a vector store from documents"""
        if not documents:
            return None
        
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key, 
                model="text-embedding-3-small"
            )
            
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None

    def setup_qa_chain(vectorstore, openai_api_key, model_name="gpt-3.5-turbo", temperature=0):
        """Set up the QA chain for financial document analysis"""
        if not vectorstore:
            return None
        
        # Create a retriever with contextual compression
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}  # Increased from 3 to 7 for better context coverage
        )
        
        # Set up LLM for compression and QA
        llm = ChatOpenAI(
            temperature=temperature, 
            model=model_name,
            openai_api_key=openai_api_key
        )
        
        # Use LLM to extract the most relevant parts of retrieved documents
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # SEC 10-Q specific prompt template - optimized for speed
        template = """You are a helpful assistant.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Always justify your answer with specific data from the documents when available.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Set up the chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa

    def highlight_query_terms(text, query):
        """Highlight query terms in text"""
        import re
        
        # Get meaningful query terms (3+ characters)
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        # Replace each term with highlighted version, case-insensitive
        highlighted_text = text
        for term in query_terms:
            pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
            highlighted_text = pattern.sub(r'<span class="highlight">\1</span>', highlighted_text)
        
        return highlighted_text

    def evaluate_response(query, context, response, api_key):
        """Evaluate response quality using Cleanlab TLM"""
        if not has_cleanlab:
            return {"error": "Cleanlab modules not available. Install with pip install cleanlab-tlm cleanlab-codex"}
        
        try:
            # Get API keys from session state
            cleanlab_tlm_key = st.session_state.CLEANLAB_TLM_API_KEY
            codex_api_key = st.session_state.CODEX_API_KEY
            codex_project_key = st.session_state.CODEX_PROJECT_KEY
            
            if not cleanlab_tlm_key or not codex_api_key or not codex_project_key:
                return {"error": "Missing Cleanlab TLM API key, Codex API key, or Codex Project Key. Please set them in the sidebar."}
            
            # Set the API keys in the environment
            os.environ["CLEANLAB_TLM_API_KEY"] = cleanlab_tlm_key
            os.environ["CODEX_API_KEY"] = codex_api_key
            
            # Import the Client class
            from cleanlab_codex.client import Client
            
            # Create a client and get the project
            codex_client = Client()
            
            # Set up custom evaluation criteria for financial document analysis
            evals = get_default_evals()
            
            # Remove query_ease from evals
            evals = [eval for eval in evals if eval.name != "query_ease"]
            
            # Add temporal context evaluation
            evals.append(
                Eval(
                    name="temporal_context_awareness",
                    criteria="""Assess whether the AI Assistant Response properly acknowledges and handles the temporal context of the information provided in the Context.

                    A good response should:
                    - Explicitly mention the time period or date of the information when relevant
                    - Use appropriate temporal qualifiers (e.g., "as of [date]", "in [year]", "during [period]")
                    - Avoid presenting historical information as current without qualification
                    - Acknowledge if the information might be outdated
                    - Use appropriate tense when discussing past information
                    - Indicate if more recent information might be available

                    For example:
                    - "The revenue was $50M" is less temporally aware than "The revenue was $50M as of Q2 2022"
                    - "The company has 100 employees" is less temporally aware than "The company had 100 employees in 2021"
                    - "The CEO is John Smith" is less temporally aware than "John Smith was the CEO as of the 2021 report"

                    A response should be considered good when it properly contextualizes the temporal nature of the information it presents.
                    """,
                    query_identifier="User Query",
                    context_identifier="Context",
                    response_identifier="AI Assistant Response",
                )
            )
            
            # Print debug information
            print(f"Using TLM API key: {cleanlab_tlm_key[:5]}...")
            print(f"Using Codex API key: {codex_api_key[:5]}...")
            print(f"Using Codex Project Key: {codex_project_key[:5]}...")
            
            validator = Validator(
                codex_access_key=codex_project_key,  # Use the project key here
                tlm_api_key=cleanlab_tlm_key,
                bad_response_thresholds={
                    "trustworthiness": 0.7,
                    "response_helpfulness": 0.7,
                    "response_groundedness": 0.7,
                    "context_sufficiency": 0.7,
                    "temporal_context_awareness": 0.7,
                },
                trustworthy_rag_config={
                    "quality_preset": "base",
                    "options": {
                        "model": "gpt-4o-mini",
                        "log": ["explanation"],
                    },
                    "evals": evals,
                }
            )
            
            print("Validator created successfully")
            
            results = validator.validate(
                query=query,
                context=context,
                response=response
            )
            
            return results
        except Exception as e:
            print(f"Error in evaluate_response: {str(e)}")
            return {"error": str(e)}

    def create_eval_visualization(eval_results):
        """Create visualization for evaluation results"""
        if "error" in eval_results:
            return None
        
        # Extract metrics and scores from nested structure
        metrics = []
        scores = []
        statuses = []
        explanations = []
        colors = []
        
        # Define the metrics we want to track (excluding query_ease)
        target_metrics = [
            "context_sufficiency",
            "response_groundedness",
            "response_helpfulness",
            "trustworthiness",
            "self_containedness",
            "temporal_context_awareness"
        ]
        
        for metric in target_metrics:
            if metric in eval_results:
                value = eval_results[metric]
                if isinstance(value, dict) and "score" in value:
                    metrics.append(metric.replace("_", " ").title())
                    scores.append(value["score"])
                    status = "Over Threshold" if value["score"] >= 0.5 else "Under Threshold"
                    statuses.append(status)
                    explanations.append(value.get("log", {}).get("explanation", ""))
                    # Set color based on threshold
                    colors.append("green" if value["score"] >= 0.5 else "red")
        
        if not metrics:
            return None
        
        # Create DataFrame
        df = pd.DataFrame({
            "Metric": metrics,
            "Score": scores,
            "Status": statuses,
            "Explanation": explanations,
            "Color": colors
        })
        
        # Create horizontal bar chart
        fig = px.bar(
            df,
            x="Score",
            y="Metric",
            orientation='h',
            range_x=[0, 1],
            color="Color",
            color_discrete_map={"green": "#2ecc71", "red": "#e74c3c"},
            labels={"Score": "Quality Score (0-1)", "Metric": "Evaluation Metric"},
            title="Response Quality Evaluation",
            hover_data=["Status", "Explanation"]
        )
        
        # Add threshold line
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold (0.5)")
        
        # Update layout for better readability
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            yaxis=dict(tickangle=0),
            hovermode="y unified"
        )
        
        return fig

    def generate_evaluation_explanation(eval_results):
        """Generate a natural language explanation of the evaluation results with detailed self-reflection"""
        if "error" in eval_results:
            return "Error in evaluation: " + eval_results["error"]
        
        issues = []
        strengths = []
        
        # Define metrics to exclude from explanation
        excluded_metrics = ["query_ease"]
        
        for metric, value in eval_results.items():
            if metric in excluded_metrics:
                continue
            
            if isinstance(value, dict) and "score" in value:
                score = value["score"]
                explanation = value.get("log", {}).get("explanation", "")
                metric_name = metric.replace("_", " ").title()
                
                if score < 0.5:
                    # Generate detailed failure explanation based on the metric
                    if metric == "temporal_context_awareness":
                        issues.append(f"### Temporal Context Awareness Failure\n"
                                    f"The response failed to properly contextualize the temporal nature of the information. {explanation}\n\n"
                                    f"**Specific Issue**: The response presents information without proper temporal qualifiers, potentially misleading users about the currency of the data. "
                                    f"For example, if discussing financial figures or operational metrics, the response should explicitly state the time period (e.g., 'as of 2014' or 'during the reporting period').")
                    elif metric == "response_groundedness":
                        issues.append(f"### Response Groundedness Failure\n"
                                    f"The response contains claims that are not properly supported by the provided context. {explanation}\n\n"
                                    f"**Specific Issue**: The response makes assertions that cannot be verified in the source documents. "
                                    f"This could include making assumptions, drawing conclusions without evidence, or presenting information that isn't explicitly stated in the context.")
                    elif metric == "context_sufficiency":
                        issues.append(f"### Context Sufficiency Failure\n"
                                    f"The provided context is insufficient to fully answer the query. {explanation}\n\n"
                                    f"**Specific Issue**: The response attempts to answer the question with incomplete information, potentially leading to misleading or incomplete answers. "
                                    f"The system should either acknowledge the limitations of the available information or request additional context.")
                    elif metric == "trustworthiness":
                        issues.append(f"### Trustworthiness Failure\n"
                                    f"The response lacks the necessary elements to establish trust. {explanation}\n\n"
                                    f"**Specific Issue**: The response may contain unverified claims, lack proper citations, or fail to acknowledge uncertainty where appropriate. "
                                    f"This undermines the reliability of the information provided.")
                    elif metric == "self_containedness":
                        issues.append(f"### Self-Containedness Failure\n"
                                    f"The response requires additional context to be fully understood. {explanation}\n\n"
                                    f"**Specific Issue**: The response makes references to concepts or information that aren't explained within the response itself, "
                                    f"potentially leaving users confused or requiring them to seek additional information.")
                    else:
                        issues.append(f"### {metric_name} Failure\n"
                                    f"The response failed to meet the quality threshold for {metric_name}. {explanation}\n\n"
                                    f"**Specific Issue**: The response scored {score:.2f} on this metric, which is below the required threshold of 0.5. "
                                    f"This indicates that improvements are needed in how the response handles {metric_name}.")
                else:
                    strengths.append(f"- {metric_name}: {explanation or f'Score of {score:.2f} meets the quality threshold'}")
        
        explanation = []
        if issues:
            explanation.append("## Failure Explanations")
            explanation.extend(issues)
        if strengths:
            explanation.append("\n## Strengths")
            explanation.extend(strengths)
        
        return "\n".join(explanation)

    def apply_remediation_instructions(instructions):
        """Create a remediation prompt template with the given instructions"""
        remediation_template = f"""
        ====== CRITICAL REMEDIATION INSTRUCTIONS ======
        
        The following instructions are important, please follow them. These are extra instructions that end users already complained about:
        {instructions}
        
        ====== END REMEDIATION INSTRUCTIONS ======
        
        You are a helpful assistant.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Always justify your answer with specific data from the documents when available.
        
        Context: {{context}}
        
        Question: {{question}}
        
        Your response MUST incorporate the remediation instructions above. These instructions take precedence over any other considerations.
        
        Answer:"""
        
        return PromptTemplate(
            template=remediation_template,
            input_variables=["context", "question"]
        )

    def save_remediation(query, instructions, eval_results):
        """Save a successful remediation for future use"""
        st.session_state.saved_remediations[query] = {
            'instructions': instructions,
            'eval_results': eval_results,
            'timestamp': time.time()
        }

    def get_saved_remediation(query):
        """Get saved remediation for a query if it exists"""
        return st.session_state.saved_remediations.get(query)

    # Main content
    st.title("RAG with TrustworthyRAG Evals")
    st.markdown("Steps: First upload PDF documents. Then ask RAG questions. See quality evals from TrustworthyRAG.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key status section
        st.subheader("API Keys Status")
        
        # Display API key status
        openai_key_status = "✅ Present" if st.session_state.OPENAI_API_KEY else "❌ Missing"
        cleanlab_tlm_key_status = "✅ Present" if st.session_state.CLEANLAB_TLM_API_KEY else "❌ Missing"
        codex_api_key_status = "✅ Present" if st.session_state.CODEX_API_KEY else "❌ Missing"
        codex_project_key_status = "✅ Present" if st.session_state.CODEX_PROJECT_KEY else "❌ Missing"
        
        st.markdown(f"**OpenAI API Key:** {openai_key_status}")
        st.markdown(f"**Cleanlab TLM API Key:** {cleanlab_tlm_key_status}")
        st.markdown(f"**Cleanlab Codex API Key:** {codex_api_key_status}")
        st.markdown(f"**Cleanlab Codex Project Key:** {codex_project_key_status}")
        
        # Add a note about setting up secrets
        with st.expander("How to set up API keys"):
            st.markdown("""
            To set up API keys in Streamlit Cloud:
            
            1. Go to your Streamlit Cloud dashboard
            2. Select your app
            3. Go to "Settings" > "Secrets"
            4. Add the following:
            ```
            OPENAI_API_KEY = "your_openai_api_key_here"
            CLEANLAB_TLM_API_KEY = "your_cleanlab_tlm_api_key_here"
            CODEX_API_KEY = "k-xxxxxx-xxxxxx"
            CODEX_PROJECT_KEY = "sk-xxxxxx-xxxxxx"
            ```
            
            Get your keys from:
            - OpenAI API key: https://platform.openai.com/api-keys
            - TLM API key: https://tlm.cleanlab.ai/
            - Codex API key: https://codex.cleanlab.ai/ (Project settings)
            - Codex Project Key: https://codex.cleanlab.ai/ (Project settings)
            """)
        
        st.divider()
        
        # Document processing settings
        st.subheader("Document Processing")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        model_name = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
            index=0  # Default to gpt-4o-mini
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        # Reset button
        if st.button("Reset Application"):
            st.session_state.documents = []
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.eval_results = None
            st.session_state.retrieved_docs = []
            st.session_state.response = ""
            st.session_state.remediation_instructions = ""
            st.session_state.remediated_response = None
            st.session_state.remediated_eval_results = None
            st.experimental_rerun()
        
        # Clear saved remediations button
        if st.button("Clear All Saved Remediations"):
            st.session_state.saved_remediations = {}
            st.success("All saved remediations have been cleared.")

        # Debug section (you can remove this after confirming keys are loaded)
        with st.expander("Debug API Keys"):
            st.write("OPENAI_API_KEY present:", bool(st.session_state.OPENAI_API_KEY))
            st.write("CLEANLAB_TLM_API_KEY present:", bool(st.session_state.CLEANLAB_TLM_API_KEY))
            st.write("CODEX_API_KEY present:", bool(st.session_state.CODEX_API_KEY))
            st.write("CODEX_PROJECT_KEY present:", bool(st.session_state.CODEX_PROJECT_KEY))

    # Document upload section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )

    # Process uploaded files
    if uploaded_files:
        if not st.session_state.OPENAI_API_KEY:
            st.error("OpenAI API key not found in Streamlit secrets. Please add it to your app's secrets.")
        else:
            # Only process if we have new files or the API key has changed
            if not st.session_state.api_key_set or st.session_state.api_key_set != st.session_state.OPENAI_API_KEY:
                with st.spinner("Processing documents..."):
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        docs = load_pdf_file(uploaded_file)
                        all_documents.extend(docs)
                    
                    if all_documents:
                        st.session_state.documents = all_documents
                        
                        # Chunk documents
                        chunked_docs = chunk_documents(
                            all_documents, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                        )
                        
                        # Create vector store
                        vectorstore = create_vector_store(chunked_docs, st.session_state.OPENAI_API_KEY)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            
                            # Set up QA chain
                            qa_chain = setup_qa_chain(
                                vectorstore,
                                st.session_state.OPENAI_API_KEY,
                                model_name=model_name,
                                temperature=temperature
                            )
                            
                            if qa_chain:
                                st.session_state.qa_chain = qa_chain
                                st.session_state.api_key_set = st.session_state.OPENAI_API_KEY
                                st.success(f"✅ Processed {len(all_documents)} document pages with {len(chunked_docs)} chunks")

    # Document stats
    if st.session_state.documents:
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Documents Loaded", len(set([d.metadata.get('source_name', 'Unknown') for d in st.session_state.documents])))
        with col2:
            st.metric("Total Pages", len(st.session_state.documents))

        # Show document list
        with st.expander("View Loaded Documents"):
            doc_sources = {}
            for doc in st.session_state.documents:
                source = doc.metadata.get('source_name', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                
                if source not in doc_sources:
                    doc_sources[source] = []
                
                doc_sources[source].append(page)
            
            for source, pages in doc_sources.items():
                st.markdown(f"**{source}**: {len(pages)} pages")

    # Add a new section for the query form that's more compact
    st.divider()
    st.header("Ask a Question")
    with st.form("query_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your question", placeholder="Type your question here...")
        with col2:
            submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if not query:
                st.warning("Please enter a question.")
            elif not st.session_state.qa_chain:
                st.warning("Please upload documents and provide your OpenAI API key first.")
            else:
                # Clear previous remediation results when a new question is asked
                st.session_state.remediation_instructions = ""
                st.session_state.remediated_response = None
                st.session_state.remediated_eval_results = None
                
                with st.spinner("Analyzing documents..."):
                    try:
                        # Check for saved remediation
                        saved_remediation = get_saved_remediation(query)
                        if saved_remediation:
                            st.info("Found a saved remediation for this query. Applying it automatically...")
                            # Display remediation details
                            with st.expander("View Saved Remediation Details"):
                                st.markdown("**Instructions Applied:**")
                                st.markdown(saved_remediation['instructions'])
                            
                            # Create a new QA chain with the remediation prompt
                            remediation_prompt = apply_remediation_instructions(saved_remediation['instructions'])
                            # Create a new QA chain using the same components as the original
                            remediation_qa = RetrievalQA.from_chain_type(
                                llm=ChatOpenAI(
                                    temperature=0,
                                    model=model_name,
                                    openai_api_key=st.session_state.OPENAI_API_KEY
                                ),
                                chain_type="stuff",
                                retriever=st.session_state.qa_chain.retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": remediation_prompt}
                            )
                            output = remediation_qa(query)
                        else:
                            output = st.session_state.qa_chain(query)
                        
                        # Store response and retrieved docs
                        st.session_state.response = output['result']
                        st.session_state.retrieved_docs = output['source_documents']
                        
                        # Show success
                        st.success("Analysis complete!")
                        
                        # Evaluate if API key provided
                        if st.session_state.CLEANLAB_TLM_API_KEY:
                            with st.spinner("Evaluating response quality..."):
                                eval_results = evaluate_response(
                                    query=query,
                                    context=str(output['source_documents']),
                                    response=output['result'],
                                    api_key=st.session_state.CLEANLAB_TLM_API_KEY
                                )
                                st.session_state.eval_results = eval_results
                                
                    except Exception as e:
                        st.error(f"Error processing query: {e}")

    # Display results if available
    if st.session_state.response:
        st.divider()
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            # Display the query response
            st.header("Original RAG Response")
            st.markdown(st.session_state.response)
            
            # Display retrieved documents
            st.subheader("Retrieved Context")
            if st.session_state.retrieved_docs:
                for i, doc in enumerate(st.session_state.retrieved_docs):
                    with st.expander(f"Document {i+1}: {doc.metadata.get('source_name', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})"):
                        # Add chunk metadata
                        st.markdown(f'<div class="chunk-header">Chunk {i+1} of {len(st.session_state.retrieved_docs)}</div>', unsafe_allow_html=True)
                        
                        # Highlight query terms if query available
                        highlighted_content = highlight_query_terms(doc.page_content, query) if 'query' in locals() else doc.page_content
                        st.markdown(f'<div class="document-preview">{highlighted_content}</div>', unsafe_allow_html=True)
                        
                        # Show chunk metadata
                        st.markdown(f'<div class="small-text">Chunk length: {len(doc.page_content)} characters</div>', unsafe_allow_html=True)
        
        with col2:
            # Display evaluation results in a more compact format
            st.header("Quality Evaluation")
            
            if not st.session_state.CLEANLAB_TLM_API_KEY:
                st.info("Enter a Cleanlab TLM API key in the sidebar to enable response evaluation.")
            elif not st.session_state.eval_results:
                st.info("Submit a query to see evaluation results.")
            elif "error" in st.session_state.eval_results:
                st.error(f"Evaluation error: {st.session_state.eval_results['error']}")
            else:
                # Create visualization
                eval_viz = create_eval_visualization(st.session_state.eval_results)
                if eval_viz:
                    st.plotly_chart(eval_viz, use_container_width=True)
                    
                    # Display numerical scores
                    st.subheader("Numerical Scores")
                    for metric, value in st.session_state.eval_results.items():
                        if isinstance(value, dict) and "score" in value:
                            score = value["score"]
                            metric_name = metric.replace("_", " ").title()
                            st.metric(metric_name, f"{score:.2f}")
                    
                    # Generate and display natural language explanation in a more compact format
                    explanation = generate_evaluation_explanation(st.session_state.eval_results)
                    with st.expander("View Evaluation Details"):
                        st.markdown(explanation)
                    
                    # Display remediated evaluation if available
                    if st.session_state.remediated_eval_results:
                        st.divider()
                        st.subheader("Remediated Evaluation")
                        remediated_eval_viz = create_eval_visualization(st.session_state.remediated_eval_results)
                        if remediated_eval_viz:
                            st.plotly_chart(remediated_eval_viz, use_container_width=True)
                            
                            # Display numerical scores for remediated results
                            st.subheader("Remediated Numerical Scores")
                            for metric, value in st.session_state.remediated_eval_results.items():
                                if isinstance(value, dict) and "score" in value:
                                    score = value["score"]
                                    metric_name = metric.replace("_", " ").title()
                                    st.metric(metric_name, f"{score:.2f}")
                            
                            # Generate and display natural language explanation in a more compact format
                            remediated_explanation = generate_evaluation_explanation(st.session_state.remediated_eval_results)
                            with st.expander("View Remediated Evaluation Details"):
                                st.markdown(remediated_explanation)
        
        with col3:
            # Remediation Panel
            st.header("Remediation Options")
            
            # Remediation options
            remediation_option = st.radio(
                "Choose a remediation approach:",
                ["Add correct answer", "Teach the AI", "Custom instructions"]
            )
            
            # Get the appropriate instructions based on selected option
            if remediation_option == "Add correct answer":
                correct_answer = st.text_area("Enter the correct answer", key="correct_answer")
                if correct_answer:
                    st.session_state.remediation_instructions = f"Here is the correct answer: {correct_answer}"
            elif remediation_option == "Teach the AI":
                teaching_instructions = st.text_area(
                    "How would you like the AI to improve?",
                    placeholder="Example: 'Make sure to cite the date of the information when discussing financial figures'",
                    key="teaching_instructions"
                )
                if teaching_instructions:
                    st.session_state.remediation_instructions = teaching_instructions
            else:  # Custom instructions
                custom_instructions = st.text_area(
                    "Enter specific instructions for improvement",
                    placeholder="Example: 'The response should explicitly mention the time period for all financial data'",
                    key="custom_instructions"
                )
                if custom_instructions:
                    st.session_state.remediation_instructions = custom_instructions
            
            # Test remediation button
            if st.button("Test Remediation"):
                with st.spinner("Applying remediation..."):
                    # Create a new QA chain with the remediation prompt
                    remediation_prompt = apply_remediation_instructions(st.session_state.remediation_instructions)
                    # Create a new QA chain using the same components as the original
                    remediation_qa = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(
                            temperature=0,
                            model=model_name,
                            openai_api_key=st.session_state.OPENAI_API_KEY
                        ),
                        chain_type="stuff",
                        retriever=st.session_state.qa_chain.retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": remediation_prompt}
                    )
                    output = remediation_qa(query)
                    
                    # Store remediated response
                    st.session_state.remediated_response = output['result']
                    
                    # Evaluate remediated response
                    if st.session_state.CLEANLAB_TLM_API_KEY:
                        eval_results = evaluate_response(
                            query=query,  # Use original query for evaluation
                            context=str(output['source_documents']),
                            response=output['result'],
                            api_key=st.session_state.CLEANLAB_TLM_API_KEY
                        )
                        st.session_state.remediated_eval_results = eval_results
            
            # Show save button if user has entered any remediation instructions
            if st.session_state.remediation_instructions:
                if st.button("Save Remediation"):
                    save_remediation(
                        query=query,
                        instructions=st.session_state.remediation_instructions,
                        eval_results={}  # Empty dict since we don't have eval results yet
                    )
                    st.success("Remediation saved!")
            
            # Display remediated response if available
            if st.session_state.remediated_response:
                st.divider()
                st.subheader("Remediated Response")
                st.markdown(st.session_state.remediated_response)

    # Footer
    st.markdown("""
    <div class="small-text" style="text-align:center; margin-top: 30px;">
        SEC 10-Q Analysis App - Built with Streamlit, LangChain, and Cleanlab TLM
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
