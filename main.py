
from private_gpt.server.embeddings.embeddings_router import (
    EmbeddingsBody,
    EmbeddingsResponse,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

    assert response.status_code == 200
    embedding_response = EmbeddingsResponse.model_validate(response.json())
    assert len(embedding_response.data) > 0
    assert len(embedding_response.data[0].embedding) > 0server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

ui:
  enabled: true
  path: /

llm:
  mode: sagemaker

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}import argparse
import logging
from pathlib import Path

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

logger = logging.getLogger(__name__)

ingest_service = root_injector.get(IngestService)

parser = argparse.ArgumentParser(prog="ingest_folder.py")
parser.add_argument("folder", help="Folder to ingest")
parser.add_argument(
    "--watch",
    help="Watch for changes",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--log-file",
    help="Optional path to a log file. If provided, logs will be written to this file.",
    type=str,
    default=None,
)
args = parser.parse_args()

# Set up logging to a file if a path is provided
if args.log_file:
    file_handler = logging.FileHandler(args.log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)


total_documents = 0
current_document_count = 0


def count_documents(folder_path: Path) -> None:
    global total_documents
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            total_documents += 1
        elif file_path.is_dir():
            count_documents(file_path)

)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

    assert response.status_code == 200
    embedding_response = EmbeddingsResponse.model_validate(response.json())
    assert len(embedding_response.data) > 0
    assert len(embedding_response.data[0].embedding) > 0server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

ui:
  enabled: true
  path: /

    EmbeddingsResponse,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

    assert response.status_code == 200
    embedding_response = EmbeddingsResponse.model_validate(response.json())
    assert len(embedding_response.data) > 0
    assert len(embedding_response.data[0].embedding) > 0server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

ui:
  enabled: true
  path: /

llm:
  mode: sagemaker

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}import argparse
import logging
from pathlib import Path

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

logger = logging.getLogger(__name__)

ingest_service = root_injector.get(IngestService)

parser = argparse.ArgumentParser(prog="ingest_folder.py")
parser.add_argument("folder", help="Folder to ingest")
parser.add_argument(
    "--watch",
    help="Watch for changes",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--log-file",
    help="Optional path to a log file. If provided, logs will be written to this file.",
    type=str,
    default=None,
from private_gpt.server.embeddings.embeddings_router import (
    EmbeddingsBody,
    EmbeddingsResponse,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

    assert response.status_code == 200
    embedding_response = EmbeddingsResponse.model_validate(response.json())
    assert len(embedding_response.data) > 0
    assert len(embedding_response.data[0].embedding) > 0server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

ui:
  enabled: true
  path: /

llm:
  mode: sagemaker

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}server:
  env_name: ${APP_ENV:prod}
  port: ${PORT:8001}

data:
  local_data_folder: local_data/private_gpt

ui:
  enabled: true
  path: /

llm:
  mode: mock

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf
  embedding_hf_model_name: BAAI/bge-small-en-v1.5

sagemaker:
  llm_endpoint_name: huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140
  embedding_endpoint_name: huggingface-pytorch-inference-2023-11-03-07-41-36-479

openai:
  api_key: ${OPENAI_API_KEY:}import argparse
import logging
from pathlib import Path

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

logger = logging.getLogger(__name__)

ingest_service = root_injector.get(IngestService)

parser = argparse.ArgumentParser(prog="ingest_folder.py")
parser.add_argument("folder", help="Folder to ingest")
parser.add_argument(
