import os
import json
import asyncio
import numpy as np
import boto3
from typing import List, Optional, Union, Any
from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class BedrockEncoder(BaseEncoder):
    client: Any  # Use 'Any' to avoid type checking issues with boto3.client
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    model_id: str = "amazon.titan-embed-text-v1"
    normalize: bool = False
    type: str = "bedrock"

    def __init__(
        self,
        name: Optional[str] = None,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        model_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        normalize: bool = False,
        score_threshold: float = 0.82,
    ):
        if name is None:
            name = "amazon.titan-embed-text-v1"  # Set the default name directly
        super().__init__(name=name, score_threshold=score_threshold)
        self.region_name = region_name or os.getenv("AWS_REGION")
        self.credentials_profile_name = credentials_profile_name
        self.endpoint_url = endpoint_url
        self.model_id = model_id or "amazon.titan-embed-text-v1"
        self.normalize = normalize

        # Initialize the boto3 client during object instantiation
        session = boto3.Session(profile_name=self.credentials_profile_name)
        self.client = session.client(
            "bedrock-runtime",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )

    def _embedding_func(self, text: str) -> List[float]:
        text = text.replace(os.linesep, " ")
        input_body = {"inputText": text}
        response = self.client.invoke_model(
            body=json.dumps(input_body),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")

    def _normalize_vector(self, embeddings: List[float]) -> List[float]:
        emb = np.array(embeddings)
        norm_emb = emb / np.linalg.norm(emb)
        return norm_emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            embedding = self._embedding_func(text)
            if self.normalize:
                embedding = self._normalize_vector(embedding)
            results.append(embedding)
        return results

    def embed_query(self, text: str) -> List[float]:
        embedding = self._embedding_func(text)
        if self.normalize:
            embedding = self._normalize_vector(embedding)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        async_tasks = [self.aembed_query(text) for text in texts]
        results = await asyncio.gather(*async_tasks)
        return results

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """
        Converts a list of documents into their embeddings.

        Args:
            docs: List of documents to be embedded.

        Returns:
            List of embeddings, each embedding is a list of floats.
        """
        return self.embed_documents(docs)
