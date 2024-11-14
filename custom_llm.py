import asyncio
import json
import logging
import aiohttp
import tiktoken
from graphrag.llm.base.base_llm import BaseLLM
from typing import List, Dict, Any, Optional, AsyncGenerator

log = logging.getLogger(__name__)

class CustomLLM(BaseLLM[List[Dict[str, str]], str]):
    """Custom LLM implementation using TGI"""
    def __init__(self, tgi_url: str = "http://<ip-address>:<port>"):
        super().__init__()
        self.tgi_url = tgi_url
        self._on_error = None
        self.logger = logging.getLogger(__name__)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any
    ) -> str:
        """Synchronously generate text."""
        return await self.agenerate(messages, **kwargs)

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        streaming: bool = False,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate text."""
        try:
            result = await self._execute_llm(
                messages,
                streaming=streaming,
                callbacks=callbacks,
                model_parameters=kwargs,
            )
            return result or ""
        except Exception as e:
            self.logger.error(f"Error in agenerate: {e}", exc_info=True)
            return ""

    async def _execute_llm(
        self,
        input: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Optional[str]:
        """Execute LLM with the given input."""
        try:
            # Format the messages into a single prompt
            prompt = "\\\\n".join(
                f"{msg.get('role', 'user').upper()}: {msg['content']}"
                for msg in input
            )
            prompt += "\\\\nASSISTANT:"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.95),
                    "do_sample": True,
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0)
                }
            }

            self.logger.debug(f"Sending request to LLM with payload: {json.dumps(payload)}")

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{self.tgi_url}/generate_stream",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30  # Add timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            self.logger.error(f"Error from LLM server (status {response.status}): {error_text}")
                            return None

                        full_response = ""
                        try:
                            async for line in response.content:
                                if line:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith("data: "):
                                        try:
                                            token_data = json.loads(line[6:])
                                            if 'token' in token_data and 'text' in token_data['token']:
                                                token = token_data['token']['text']
                                                full_response += token
                                        except json.JSONDecodeError as e:
                                            self.logger.warning(f"Failed to decode token: {e}")
                                            continue

                        except Exception as e:
                            self.logger.error(f"Error processing stream: {e}", exc_info=True)
                            if full_response:  # Return partial response if available
                                return full_response.strip()
                            return None

                except aiohttp.ClientError as e:
                    self.logger.error(f"Network error connecting to LLM: {e}")
                    return None
                except asyncio.TimeoutError:
                    self.logger.error("Request to LLM timed out")
                    return None

            return full_response.strip() if full_response else None

        except Exception as e:
            self.logger.error(f"Error executing LLM: {e}", exc_info=True)
            return None

    async def _stream_execute_llm(
        self,
        messages: List[Dict[str, str]],
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream execute LLM with the given input."""
        response = await self._execute_llm(messages, streaming=True, callbacks=callbacks, **kwargs)
        if response:
            yield response

    async def astream_generate(
        self,
        messages: List[Dict[str, str]],
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream generate text asynchronously."""
        async for token in self._stream_execute_llm(messages, callbacks=callbacks, **kwargs):
            yield token
