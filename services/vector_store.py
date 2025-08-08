# #vector_store.py
# import os
# from pinecone import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from config.settings import settings
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from groq import Groq, APIError
# # Initialize Pinecone
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# index = pc.Index(settings.PINECONE_INDEX_NAME)

# # # Initialize HuggingFace embedding model
# # embedding_model = HuggingFaceEmbeddings(
# #     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # )

# embedding_model = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# # llm = HuggingFaceEndpoint(
# #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
# #     task="text-generation",
# #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
# #     temperature=0.0,
# #     top_k=1,
# #     top_p=1.0,
# #     do_sample=False,
# #     repetition_penalty=1.0,
# # )
# # model = ChatHuggingFace(llm=llm)
#  # The user's prompt. You can change this to test different inputs.
# user_prompt = "Explain the importance of low-latency LLMs in 100 words."
# completion = client.chat.completions.create(
#     model="llama3-70b-8192",
#     messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a helpful assistant."
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt,
#                 }],
#     temperature=0.0,
#     top_p=1.0,
#     stream=True  # or False if you want the full response at once
# )


# def split_text(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_text(text)

# async def embed_and_upsert(chunks: list[str], namespace: str):
#     print(f"Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
#     try:
#         batch_size = 100
#         total_batches = (len(chunks) + batch_size - 1) // batch_size
#         total_inserted = 0

#         print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
#             current_batch_number = (i // batch_size) + 1
#             print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

#             embeddings = embedding_model.embed_documents(batch)

#             vectors = []
#             for j, embedding in enumerate(embeddings):
#                 text = batch[j]
#                 metadata = {
#                     "text": text,
#                     "section": "unknown",
#                     "page": -1,
#                     "source": "",
#                     "type": "paragraph",
#                 }

#                 vectors.append({
#                     "id": f"{namespace}_{i + j}",
#                     "values": embedding,
#                     "metadata": metadata
#                 })

#             print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
#             response = index.upsert(vectors=vectors, namespace=namespace)
#             print(f"✅ Upsert for batch {current_batch_number} completed. Response: {response}")
#             total_inserted += len(vectors)

#         return {"status": "success", "inserted": total_inserted}

#     except Exception as e:
#         print(f"❌ Error in embed_and_upsert: {e}")
#         return {"status": "error", "error": str(e)}

# async def retrieve_from_kb(input_params):
#     try:
#         query = input_params.get("query", "")
#         agent_id = input_params.get("agent_id", "")
#         top_k = input_params.get("top_k", 5)

#         if not query:
#             return {"chunks": [], "status": "error", "message": "Query is required"}
#         if not agent_id:
#             return {"chunks": [], "status": "error", "message": "Agent ID is required"}

#         # Get embedding for query
#         query_vector = embedding_model.embed_query(query)

#         # Search in Pinecone using the vector
#         results = index.query(
#             vector=query_vector,
#             namespace=agent_id,
#             top_k=top_k,
#             include_metadata=True
#         )

#         content_blocks = []
#         for match in results.matches:
#             score = match.score
#             if score > 0.0:
#                 metadata = match.metadata or {}
#                 text = metadata.get("text", "")
#                 if text:
#                     content_blocks.append(text)

#         return {"chunks": content_blocks}

#     except Exception as e:
#         print(f"Error in retrieve_from_kb: {e}")
#         return {"chunks": [], "status": "error", "error": str(e)}

# # Function routing
# FUNCTION_HANDLERS = {
#     "retrieve_from_kb": retrieve_from_kb
# }

# FUNCTION_DEFINITIONS = [
#     {
#         "name": "retrieve_from_kb",
#         "description": "Retrieves top-k chunks from the knowledge base using a query and agent_id (namespace).",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The user's search query."
#                 },
#                 "agent_id": {
#                     "type": "string",
#                     "description": "The namespace or agent ID to search in."
#                 },
#                 "top_k": {
#                     "type": "integer",
#                     "description": "Number of top results to return.",
#                     "default": 3
#                 }
#             },
#             "required": ["query", "agent_id"]
#         }
#     }
# ]


#vector_store.py
import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from config.settings import settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from groq import Groq, APIError

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)

# # Initialize HuggingFace embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     temperature=0.0,
#     top_k=1,
#     top_p=1.0,
#     do_sample=False,
#     repetition_penalty=1.0,
# )
# model = ChatHuggingFace(llm=llm)
 # The user's prompt. You can change this to test different inputs.
user_prompt = "Explain the importance of low-latency LLMs in 100 words."
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }],
    temperature=0.0,
    top_p=1.0,
    stream=True  # or False if you want the full response at once
)


def split_text(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

async def embed_and_upsert(chunks: list[str], namespace: str, document_metadata: dict = None):
    """
    Enhanced function to embed and upsert chunks with comprehensive metadata
    
    Args:
        chunks: List of text chunks to embed
        namespace: Namespace for the vector store
        document_metadata: Dictionary containing document-level metadata
    """
    print(f"Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
    try:
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        total_inserted = 0

        print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

        # Default metadata if none provided
        if document_metadata is None:
            document_metadata = {}

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            current_batch_number = (i // batch_size) + 1
            print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

            embeddings = embedding_model.embed_documents(batch)

            vectors = []
            for j, embedding in enumerate(embeddings):
                text = batch[j]
                
                # Enhanced metadata structure based on your example
                metadata = {
                    "text": text,
                    "chunk_index": i + j,
                    # Document identification
                    "author": document_metadata.get("author", ""),
                    "title": document_metadata.get("title", ""),
                    "subject": document_metadata.get("subject", ""),
                    "keywords": document_metadata.get("keywords", ""),
                    
                    # File information
                    "file_path": document_metadata.get("file_path", ""),
                    "source_file": document_metadata.get("source_file", ""),
                    "source": document_metadata.get("source", ""),
                    "format": document_metadata.get("format", ""),
                    "producer": document_metadata.get("producer", ""),
                    
                    # Page and section info
                    "page": document_metadata.get("page", -1),
                    "total_pages": document_metadata.get("total_pages", ""),
                    "section": document_metadata.get("section", "unknown"),
                    "type": document_metadata.get("type", "paragraph"),
                    
                    # Timestamps
                    "creationDate": document_metadata.get("creationDate", ""),
                    "modDate": document_metadata.get("modDate", ""),
                    "creation_date": document_metadata.get("creation_date", ""),
                    "mod_date": document_metadata.get("mod_date", ""),
                    
                    # Additional fields
                    "creator": document_metadata.get("creator", ""),
                    "trapped": document_metadata.get("trapped", ""),
                    
                    # Custom fields for your use case
                    "document_id": document_metadata.get("document_id", ""),
                    "category": document_metadata.get("category", ""),
                    "language": document_metadata.get("language", ""),
                    "content_type": document_metadata.get("content_type", "text"),
                }

                vectors.append({
                    "id": f"{namespace}_{i + j}",
                    "values": embedding,
                    "metadata": metadata
                })

            print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
            response = index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅ Upsert for batch {current_batch_number} completed. Response: {response}")
            total_inserted += len(vectors)

        return {"status": "success", "inserted": total_inserted}

    except Exception as e:
        print(f"❌ Error in embed_and_upsert: {e}")
        return {"status": "error", "error": str(e)}

async def retrieve_from_kb(input_params):
    """
    Enhanced retrieval function with metadata filtering capabilities
    """
    try:
        query = input_params.get("query", "")
        agent_id = input_params.get("agent_id", "")
        top_k = input_params.get("top_k", 5)
        metadata_filter = input_params.get("filter", {})  # Added metadata filtering

        if not query:
            return {"chunks": [], "status": "error", "message": "Query is required"}
        if not agent_id:
            return {"chunks": [], "status": "error", "message": "Agent ID is required"}

        # Get embedding for query
        query_vector = embedding_model.embed_query(query)

        # Search in Pinecone using the vector with optional metadata filtering
        search_params = {
            "vector": query_vector,
            "namespace": agent_id,
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            search_params["filter"] = metadata_filter

        results = index.query(**search_params)

        content_blocks = []
        detailed_results = []  # Enhanced results with full metadata
        
        for match in results.matches:
            score = match.score
            if score > 0.0:
                metadata = match.metadata or {}
                text = metadata.get("text", "")
                if text:
                    content_blocks.append(text)
                    
                    # Enhanced result with metadata for potential use
                    detailed_results.append({
                        "text": text,
                        "score": score,
                        "metadata": {
                            "author": metadata.get("author", ""),
                            "title": metadata.get("title", ""),
                            "page": metadata.get("page", -1),
                            "source": metadata.get("source", ""),
                            "file_path": metadata.get("file_path", ""),
                            "subject": metadata.get("subject", ""),
                            "keywords": metadata.get("keywords", ""),
                            "creation_date": metadata.get("creationDate", ""),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "section": metadata.get("section", "unknown"),
                            "type": metadata.get("type", "paragraph")
                        }
                    })

        return {
            "chunks": content_blocks,
            "detailed_results": detailed_results,  # Added for enhanced retrieval info
            "status": "success"
        }

    except Exception as e:
        print(f"Error in retrieve_from_kb: {e}")
        return {"chunks": [], "status": "error", "error": str(e)}

# Function routing
FUNCTION_HANDLERS = {
    "retrieve_from_kb": retrieve_from_kb
}

FUNCTION_DEFINITIONS = [
    {
        "name": "retrieve_from_kb",
        "description": "Retrieves top-k chunks from the knowledge base using a query and agent_id (namespace) with optional metadata filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query."
                },
                "agent_id": {
                    "type": "string",
                    "description": "The namespace or agent ID to search in."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 3
                },
                "filter": {
                    "type": "object",
                    "description": "Optional metadata filter for refined search (e.g., {'author': 'Newton', 'format': 'PDF'}).",
                    "default": {}
                }
            },
            "required": ["query", "agent_id"]
        }
    }
]