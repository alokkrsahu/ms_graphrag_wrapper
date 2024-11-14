import logging
import pandas as pd
from graphrag.query.context_builder.builders import LocalContextBuilder, ContextBuilderResult
from graphrag.query.context_builder.conversation_history import ConversationHistory
from typing import List, Dict, Any
from typing import Optional

log = logging.getLogger(__name__)

class MDContextBuilder(LocalContextBuilder):
    """Custom context builder for MD files"""
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.df = pd.DataFrame([
            {
                "text": doc["text"],
                **doc["metadata"]
            }
            for doc in documents
        ])

    def build_context(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> ContextBuilderResult:
        """Build context from MD documents"""
        try:
            query_terms = query.lower().split()

            matched_docs = []
            for doc in self.documents:
                text = doc["text"].lower()
                score = sum(1 for term in query_terms if term in text)
                if score > 0:
                    matched_docs.append((doc, score))

            matched_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = matched_docs[:5] if matched_docs else self.documents[:5]

            # Fixed context text formatting
            context_text = "\\n\\n---\\n\\n".join(
                f'''Source: {doc['metadata']['source']}
                Relevance Score: {score}
                Content: {doc['text'][:1000]}'''  # Limit text size per doc
                for doc, score in top_docs
            )

            context_records = {
                "documents": pd.DataFrame([
                    {
                        "text": doc["text"],
                        "score": score,
                        **doc["metadata"]
                    }
                    for doc, score in top_docs
                ])
            }

            system_message = f'''You are a helpful assistant. Use the following context to answer the user's question.
            If you cannot find relevant information in the context, use your general knowledge to provide a helpful response.

            Context:
            {context_text}

            Answer the question based on the above context and your knowledge. If using information from the context,
            cite the source in your response.'''

            return ContextBuilderResult(
                context_chunks=system_message,
                context_records=context_records,
                llm_calls=0,
                prompt_tokens=len(system_message.split()),
                output_tokens=0
            )
        except Exception as e:
            log.error(f"Error building context: {e}", exc_info=True)
            return ContextBuilderResult(
                context_chunks="",
                context_records={"documents": pd.DataFrame()},
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0
            )
