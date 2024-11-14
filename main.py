import asyncio
import logging, os, time

from custom_llm import CustomLLM
from md_context_builder import MDContextBuilder
from local_search import LocalSearch
from file_processor import FileProcessor
import tiktoken
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.storage_config import StorageConfig
from graphrag.config.models.text_embedding_config import TextEmbeddingConfig
from graphrag.config.models.llm_parameters import LLMParameters

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # Base directory containing MD files
    base_dir = ""
    data_dir = "./graphrag_data"

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    print("Initializing GraphRAG...")
    try:
        # Initialize llm parameters
        llm_params = LLMParameters(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2000,
            repetition_penalty=1.0
        )

        # Initialize configuration
        config = GraphRagConfig(
            root_dir=data_dir,
            storage=StorageConfig(
                type="file",
                root_dir=data_dir
            ),
            embeddings=TextEmbeddingConfig(
                model="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=1024,
                chunk_overlap=100
            ),
            llm=llm_params
        )

        # Initialize file processor and clean filenames
        print("Cleaning filenames...")
        FileProcessor.sanitize_directory(base_dir)

        # Read all MD files
        print("Reading MD files...")
        documents = FileProcessor.read_md_files(base_dir)
        print(f"Found {len(documents)} documents")

        # Initialize components
        context_builder = MDContextBuilder(documents)
        llm = CustomLLM()

        # Test LLM connection
        print("Testing LLM connection...")
        test_message = [{"role": "user", "content": "Hello, are you working?"}]
        test_response = await llm.agenerate(test_message, max_tokens=20)
        if not test_response:
            raise Exception("Failed to get response from LLM. Please check the connection and server status.")
        print("LLM connection test successful!")

        # Initialize local search
        print("Initializing search components...")
        token_encoder = tiktoken.get_encoding("cl100k_base")
        search = LocalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params={"max_tokens": 1500, "temperature": 0.0}
        )

        print("\\nInitialization complete. Ready for queries!")

        # Interactive query loop
        while True:
            try:
                # Get user input
                query = input("\\nEnter your question (or 'quit' to exit): ")
                if query.lower() == 'quit':
                    break

                if not query.strip():
                    print("Please enter a valid query.")
                    continue

                logger.debug("Starting search for query: %s", query)
                print("\\nGenerating response...")

                # Create message format
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate and relevant information based on the given context."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]

                # First try direct LLM response for comparison
                print("\\nDirect LLM response:")
                try:
                    direct_response = await llm.agenerate(
                        messages,
                        max_tokens=200,
                        temperature=0.7
                    )
                    if direct_response:
                        print(f"Direct response: {direct_response}\\n")
                    else:
                        print("No direct response generated.")
                except Exception as e:
                    logger.error(f"Error getting direct response: {e}")
                    print("Failed to get direct response.")

                # Then try the full RAG search
                print("Generating RAG response...")
                try:
                    result = await search.asearch(query)

                    logger.debug("Search completed")
                    logger.debug("Response length: %d", len(result.response) if result.response else 0)
                    logger.debug("Context length: %d", len(result.context_text) if result.context_text else 0)

                    if result.response and result.response.strip():
                        print("\\nRAG Response:")
                        print(f"{result.response}")

                        if result.context_text:
                            print("\\nContext Used:")
                            print(f"{result.context_text[:500]}...")
                            print("\\n(Context truncated for display)")
                    else:
                        print("\\nNo RAG response generated.")
                        logger.warning("Empty response from RAG search")

                    # Print some stats if available
                    if hasattr(result, 'metrics'):
                        print("\\nSearch Metrics:")
                        for key, value in result.metrics.items():
                            print(f"{key}: {value}")

                except Exception as e:
                    logger.error(f"Error in RAG search: {e}", exc_info=True)
                    print(f"\\nError during RAG search: {str(e)}")

            except KeyboardInterrupt:
                print("\\nExiting due to user interrupt...")
                break

            except Exception as e:
                logger.error("Unexpected error in main loop: %s", str(e), exc_info=True)
                print(f"\\nAn unexpected error occurred: {str(e)}")
                print("The system will continue running. Please try another query.")
                continue

        print("\\nThank you for using the system!")

    except Exception as e:
        logger.error("Fatal error during initialization: %s", str(e), exc_info=True)
        print(f"Fatal Error: {str(e)}")
        print("Please ensure all components are properly installed and configured.")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
