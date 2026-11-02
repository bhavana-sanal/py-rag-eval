"""Builds concrete pipeline components from validated config objects."""

from __future__ import annotations

import os

import httpx

from rag_eval.chunking import Chunker, FixedSizeChunker, RecursiveChunker, SentenceWindowChunker
from rag_eval.config import (
    BM25Config,
    ChunkerConfig,
    DenseConfig,
    EmbedderConfig,
    EndpointConfig,
    ExperimentConfig,
    ExtractiveGeneratorConfig,
    FixedChunkerConfig,
    GeneratorConfig,
    HybridConfig,
    JudgeConfig,
    LLMGeneratorConfig,
    LsaEmbedderConfig,
    RecursiveChunkerConfig,
    SentenceTransformerEmbedderConfig,
    SentenceWindowChunkerConfig,
)
from rag_eval.embeddings import Embedder, LsaEmbedder, SentenceTransformerEmbedder
from rag_eval.generation import ExtractiveGenerator, Generator, LLMGenerator
from rag_eval.judge import JudgeCache, LLMJudge
from rag_eval.llm import ChatClient
from rag_eval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MMRReranker,
    RerankedRetriever,
    Retriever,
)


def build_chunker(config: ChunkerConfig) -> Chunker:
    match config:
        case FixedChunkerConfig(size=size, overlap=overlap):
            return FixedSizeChunker(size, overlap)
        case RecursiveChunkerConfig(max_words=max_words):
            return RecursiveChunker(max_words)
        case SentenceWindowChunkerConfig(window=window):
            return SentenceWindowChunker(window)


def build_embedder(config: EmbedderConfig) -> Embedder:
    match config:
        case LsaEmbedderConfig(dim=dim):
            return LsaEmbedder(dim)
        case SentenceTransformerEmbedderConfig(model=model):
            return SentenceTransformerEmbedder(model)


def build_retriever(name: str, experiment: ExperimentConfig) -> Retriever:
    config = experiment.retrievers[name]
    retriever: Retriever
    match config:
        case BM25Config(k1=k1, b=b):
            retriever = BM25Retriever(k1, b)
        case DenseConfig():
            retriever = DenseRetriever(build_embedder(experiment.embedder))
        case HybridConfig(members=members, rrf_k=rrf_k, depth=depth):
            retriever = HybridRetriever(
                [build_retriever(m, experiment) for m in members], rrf_k, depth
            )
    if config.mmr_lambda is not None:
        reranker = MMRReranker(build_embedder(experiment.embedder), config.mmr_lambda)
        retriever = RerankedRetriever(retriever, reranker, config.mmr_fetch_k)
    return retriever


def build_chat_client(config: EndpointConfig, http_client: httpx.Client | None) -> ChatClient:
    return ChatClient(
        config.model,
        base_url=config.base_url,
        api_key=os.environ.get(config.api_key_env),
        timeout=config.timeout,
        http_client=http_client,
    )


def build_generator(config: GeneratorConfig, http_client: httpx.Client | None = None) -> Generator:
    match config:
        case ExtractiveGeneratorConfig():
            return ExtractiveGenerator()
        case LLMGeneratorConfig(temperature=temperature):
            return LLMGenerator(build_chat_client(config, http_client), temperature)


def build_judge(config: JudgeConfig, http_client: httpx.Client | None = None) -> LLMJudge:
    return LLMJudge(build_chat_client(config, http_client), JudgeCache(config.cache_dir))
