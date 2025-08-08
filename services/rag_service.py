#rag_service
from pinecone import Pinecone
from services.vector_store import retrieve_from_kb
from services.hf_model import ask_gpt
import re
import asyncio
import os
import tempfile
import requests
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from urllib.parse import urlparse
from config.settings import settings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

def generate_namespace_from_url(url: str) -> str:
    """Generate namespace matching the embedding script logic"""
    try:
        # Parse URL to get the filename part
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1]
        
        # Remove query parameters if they're part of the filename
        if '?' in filename:
            filename = filename.split('?')[0]
            
        # Remove extension (matching your embedding script logic)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Replace non-alphanumeric characters with underscores and convert to lowercase
        # This matches exactly: re.sub(r'[^a-zA-Z0-9]+', '_', name_without_ext).strip('_').lower()
        namespace = re.sub(r'[^a-zA-Z0-9]+', '_', name_without_ext).strip('_').lower()
        
        # Ensure namespace is not empty (matching your embedding script)
        if not namespace:
            namespace = "default_namespace"
            
        return namespace
        
    except Exception as e:
        print(f"Error generating namespace: {e}")
        return "default_namespace"

def clean_metadata_for_pinecone(metadata: dict, max_total_size=2000) -> dict:
    """
    Clean metadata to ensure it's compatible with Pinecone and under size limit.
    """
    cleaned = {}
    total_size = 0

    for key, value in metadata.items():
        if value is None:
            value = "unknown"
        elif isinstance(value, (list, dict)):
            value = str(value)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)

        value_str = str(value)

        if len(value_str) > 300:  # Truncate very long values
            value_str = value_str[:300] + "..."

        total_size += len(key) + len(value_str)
        if total_size > max_total_size:
            print(f"⚠️ Skipping metadata key '{key}' to stay under limit")
            continue

        cleaned[key] = value_str

    return cleaned

def load_and_split_pdf(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Load PDF and split into chunks using RecursiveCharacterTextSplitter."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_name = os.path.basename(file_path)
    print(f"📄 Loading PDF: {file_name}")

    # Load PDF using PyMuPDFLoader
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF {file_name}: {str(e)}")

    if not docs:
        raise ValueError(f"No content found in PDF: {file_name}")

    print(f"📖 Loaded {len(docs)} pages from PDF")

    # Add source file metadata
    for doc in docs:
        doc.metadata["source_file"] = file_name
        if "page" not in doc.metadata:
            doc.metadata["page"] = "unknown"

    # Split documents using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"📚 Split into {len(chunks)} chunks")
    
    return chunks

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL to temporary file and return the file path."""
    try:
        print(f"📥 Downloading PDF from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            print(f"⚠️ Warning: Content type is {content_type}, may not be a PDF")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        print(f"✅ PDF downloaded to: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"❌ Error downloading PDF: {e}")
        raise

async def embed_pdf_to_pinecone(pdf_url: str, namespace: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process PDF from URL and upload chunks to Pinecone."""
    print(f"🚀 Starting PDF embedding for namespace: {namespace}")
    
    temp_file_path = None
    try:
        # Download PDF to temporary file
        temp_file_path = download_pdf_from_url(pdf_url)
        
        # Load and split PDF
        chunks = load_and_split_pdf(temp_file_path, chunk_size, chunk_overlap)
        
        if not chunks:
            print("❌ No chunks to process")
            return False

        # Initialize embedding model (using local sentence transformer)
        try:
            embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            )
            print("🤖 Local embedding model initialized")
        except Exception as e:
            print(f"❌ Error initializing embedding model: {str(e)}")
            return False

        # Clean metadata for all chunks
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_metadata = clean_metadata_for_pinecone(chunk.metadata)
            # Add URL to metadata
            cleaned_metadata["source_url"] = pdf_url
            cleaned_chunk = Document(
                page_content=chunk.page_content, 
                metadata=cleaned_metadata
            )
            cleaned_chunks.append(cleaned_chunk)

        # Upload to Pinecone
        try:
            print("📌 Uploading to Pinecone...")
            
            vectorstore = PineconeVectorStore.from_documents(
                documents=cleaned_chunks,
                embedding=embedding_model,
                index_name=settings.PINECONE_INDEX_NAME,
                namespace=namespace
            )
            
            print(f"✅ Successfully embedded and stored {len(chunks)} chunks")
            print(f"📍 Namespace: {namespace}")
            print(f"🔍 Index: {settings.PINECONE_INDEX_NAME}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading to Pinecone: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Error in PDF embedding process: {e}")
        return False
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"🗑️ Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"⚠️ Error cleaning up temp file: {e}")

async def list_available_namespaces() -> list[str]:
    """Helper function to list all available namespaces in the Pinecone index"""
    try:
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        return namespaces
    except Exception as e:
        print(f"Error retrieving namespaces: {e}")
        return []

async def process_documents_and_questions(pdf_url: str, questions: list[str], namespace: str = None) -> dict:
    print(f"Processing questions for PDF URL: {pdf_url}")
    
    try:
        # Step 1: Handle namespace determination
        if namespace:
            # Use provided namespace directly
            agent_id = namespace
            print(f"📂 Using provided namespace: '{agent_id}'")
        else:
            # Generate namespace using the same logic as your embedding scripts
            agent_id = generate_namespace_from_url(pdf_url)
            print(f"📂 Generated namespace: '{agent_id}'")
        
        # Debug: Check what namespaces actually exist
        try:
            stats = index.describe_index_stats()
            existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            print(f"🔍 Available namespaces: {existing_namespaces}")
            
            namespace_exists = agent_id in existing_namespaces
            
            if not namespace_exists:
                print(f"⚠️ Namespace '{agent_id}' not found in existing namespaces")
                
                if not namespace:  # Only auto-select if namespace wasn't provided
                    # Try common patterns based on your embedding scripts
                    possible_namespaces = [
                        agent_id,  # Direct match
                        "extracted_text_embedding",  # For text files
                        f"{agent_id}_pdf",  # With suffix
                        f"doc_{agent_id}",  # With prefix
                    ]
                    
                    # Also try partial matches for existing namespaces
                    for existing_ns in existing_namespaces:
                        if agent_id in existing_ns.lower() or existing_ns.lower() in agent_id:
                            possible_namespaces.append(existing_ns)
                    
                    found_namespace = None
                    for candidate in possible_namespaces:
                        if candidate in existing_namespaces:
                            found_namespace = candidate
                            break
                    
                    if found_namespace:
                        agent_id = found_namespace
                        print(f"🔄 Found matching namespace: '{agent_id}'")
                        namespace_exists = True
                    else:
                        # Namespace doesn't exist, so embed the PDF
                        print(f"📥 Namespace not found. Embedding PDF from URL...")
                        embedding_success = await embed_pdf_to_pinecone(
                            pdf_url=pdf_url,
                            namespace=agent_id,
                            chunk_size=500,
                            chunk_overlap=100
                        )
                        
                        if not embedding_success:
                            raise Exception(f"Failed to embed PDF from URL: {pdf_url}")
                        
                        print(f"✅ Successfully created namespace '{agent_id}' with PDF embeddings")
                        namespace_exists = True
                else:
                    # Provided namespace doesn't exist, embed the PDF
                    print(f"📥 Provided namespace '{namespace}' not found. Embedding PDF from URL...")
                    embedding_success = await embed_pdf_to_pinecone(
                        pdf_url=pdf_url,
                        namespace=agent_id,
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    
                    if not embedding_success:
                        raise Exception(f"Failed to embed PDF for namespace '{namespace}'")
                    
                    print(f"✅ Successfully created namespace '{agent_id}' with PDF embeddings")
                
        except Exception as e:
            if "Failed to embed PDF" in str(e):
                raise e
            print(f"Error checking namespaces: {e}")
            # Continue with the generated namespace anyway

        # Step 2: Parallel question processing with reduced concurrency
        semaphore = asyncio.Semaphore(3)  # Reduced from 10 to 3 to avoid rate limits

        async def process_question(index: int, question: str) -> tuple[int, str, str]:
            async with semaphore:
                for attempt in range(3):
                    try:
                        retrieval_input = {"query": question, "agent_id": agent_id, "top_k": 3}
                        retrieved = await retrieve_from_kb(retrieval_input)
                        retrieved_chunks = retrieved.get("chunks", [])
                        
                        if not retrieved_chunks:
                            print(f"⚠️ Q{index}: No chunks retrieved for question: {question[:50]}...")
                            return (index, question, "I couldn't find relevant information to answer this question.")

                        max_context_chars = 3000
                        context = "\n".join(retrieved_chunks)[:max_context_chars]

                        print(f"✏️ Q{index}: Processing question: {question[:50]}...")
                        answer = await ask_gpt(context, question)
                        return (index, question, answer)
                        
                    except Exception as e:
                        print(f"⚠️ Q{index}: Attempt {attempt + 1} failed with error: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                return (index, question, "Sorry, I couldn't find relevant information to answer this question.")

        print(f"🧠 Parallel processing {len(questions)} questions...")
        
        if not questions:
            return {}
            
        # Add timeout for question processing
        try:
            tasks = [asyncio.create_task(process_question(i, q)) for i, q in enumerate(questions)]
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)  # 2 minute timeout
        except asyncio.TimeoutError:
            print("⚠️ Question processing timed out")
            raise Exception("Processing timed out. Please try with fewer questions or a smaller document.")

        # Step 3: Return sorted results
        results = {q: ans for _, q, ans in sorted(responses)}
        return results
        
    except Exception as e:
        print(f"❌ Error in process_documents_and_questions: {e}")
        raise
