
from private_gpt.server.embeddings.embeddings_router import (
    EmbeddingsBody,
    EmbeddingsResponse,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

