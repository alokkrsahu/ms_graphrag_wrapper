import asyncio
import logging
from typing import List, Dict, Any
from typing import Optional
import time 

from custom_llm import CustomLLM
from md_context_builder import MDContextBuilder

log = logging.getLogger(__name__)

class LocalSearch:
    def __init__(
        self,
        llm: CustomLLM,
        context_builder: MDContextBuilder,
        token_encoder: Any,
        llm_params: Dict[str, Any]
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.token_encoder = token_encoder
        self.llm_params = llm_params
        self.logger = logging.getLogger(__name__)

    async def asearch(self, query: str) -> Any:
        try:
            # Build context
            context_result = self.context_builder.build_context(query)

            if not context_result.context_chunks:
                self.logger.warning("No context found for query")

            # Log the context for debugging
            self.logger.debug(f"Generated context: {context_result.context_chunks[:500]}...")

            # Prepare messages for LLM with more explicit system prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. Your task is to provide accurate "
                        "and relevant information based on the context provided. If the context "
                        "contains relevant information, use it in your response. If not, provide "
                        "a general response based on your knowledge.\\n\\n"
                        f"{context_result.context_chunks}"
                    )
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            # Log the full message structure
            self.logger.debug(f"Sending messages to LLM: {messages}")

            # Generate response with timeout
            self.logger.info(f"GENERATE ANSWER: {time.time()}. QUERY: {query}")

            # Add explicit parameters to the LLM call
            response = await self.llm.agenerate(
                messages,
                max_tokens=self.llm_params.get('max_tokens', 1500),
                temperature=self.llm_params.get('temperature', 0.0),
                top_p=self.llm_params.get('top_p', 0.95),
                repetition_penalty=self.llm_params.get('repetition_penalty', 1.0)
            )

            if not response:
                self.logger.error("LLM returned empty response")
                response = "I apologize, but I was unable to generate a response. Please try rephrasing your question."

            # Create detailed metrics
            metrics = {
                'context_tokens': len(self.token_encoder.encode(context_result.context_chunks)),
                'response_tokens': len(self.token_encoder.encode(response)) if response else 0,
                'matched_documents': len(context_result.context_records['documents']),
                'total_context_length': len(context_result.context_chunks),
                'query_length': len(query)
            }

            # Create result object with more detailed information
            result = type('SearchResult', (), {
                'response': response,
                'context_text': context_result.context_chunks,
                'metrics': metrics,
                'success': bool(response and response.strip()),
                'matched_docs': len(context_result.context_records['documents'])
            })

            return result

        except Exception as e:
            self.logger.error(f"Error in asearch: {e}", exc_info=True)
            # Return a result object even in case of error
            return type('SearchResult', (), {
                'response': f"An error occurred while processing your query: {str(e)}",
                'context_text': "",
                'metrics': {
                    'context_tokens': 0,
                    'response_tokens': 0,
                    'matched_documents': 0,
                    'error': str(e)
                },
                'success': False,
                'matched_docs': 0
            })
