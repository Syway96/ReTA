#!/usr/bin/env python3
"""
统一配置模块
集中管理项目中所有的 dataclass 配置定义、YAML 加载和环境变量覆盖。
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import yaml
from dacite import from_dict
from dotenv import load_dotenv


# ==================== 嵌入 & LLM 配置 ====================

@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    local_path: Optional[str] = None
    online_fallback: Optional[str] = None
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """大语言模型配置"""
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    num_predict: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_ctx: Optional[int] = None
    reasoning_effort: Optional[str] = None


# ==================== QA 系统配置 ====================

@dataclass
class RetrievalConfig:
    """检索配置"""
    k_per_store: Optional[int] = None
    total_max_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    enable_reranking: Optional[bool] = None
    dynamic_complexity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankConfig:
    """重排序配置"""
    enabled: Optional[bool] = None
    method: Optional[str] = None
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    use_cross_encoder: Optional[bool] = None
    cross_encoder_model: Optional[str] = None
    cross_encoder_local_path: Optional[str] = None


@dataclass
class PromptConfig:
    """提示词配置"""
    system_template: Optional[str] = None
    human_template: Optional[str] = None


@dataclass
class SystemConfig:
    """系统配置"""
    enable_multiple_stores: Optional[bool] = None
    show_retrieval_info: Optional[bool] = None
    log_level: Optional[str] = None
    streaming_output: Optional[bool] = None
    streaming_delay: Optional[float] = None


@dataclass
class QASystemConfig:
    """问答系统配置"""
    project_dir: Optional[str] = None
    vector_store_dir: Optional[str] = None
    data_dir: Optional[str] = None
    embedding: Optional[EmbeddingConfig] = None
    llm: Optional[LLMConfig] = None
    retrieval: Optional[RetrievalConfig] = None
    rerank: Optional[RerankConfig] = None
    prompt: Optional[PromptConfig] = None
    system: Optional[SystemConfig] = None


# ==================== 数据处理配置 ====================

@dataclass
class MarkdownChunkerConfig:
    """Markdown 分块器配置"""
    enabled: Optional[bool] = None
    max_chunk_chars: Optional[int] = None
    min_chunk_chars: Optional[int] = None
    preserve_headings: Optional[bool] = None
    combine_small_paragraphs: Optional[bool] = None
    heading_level: Optional[int] = None
    include_code_blocks: Optional[bool] = None
    code_block_min_lines: Optional[int] = None


@dataclass
class CleanerConfig:
    """Markdown 文本清洗配置"""
    remove_empty_lines: Optional[bool] = None
    normalize_whitespace: Optional[bool] = None
    remove_metadata: Optional[bool] = None


@dataclass
class StreamingConfig:
    """流式处理配置"""
    enabled: Optional[bool] = None
    max_file_size_mb: Optional[int] = None


@dataclass
class ProcessingConfig:
    """完整数据处理配置"""
    output_dir: Optional[str] = None
    markdown_chunker: Optional[MarkdownChunkerConfig] = None
    cleaner: Optional[CleanerConfig] = None
    streaming: Optional[StreamingConfig] = None
    skip_existing: Optional[bool] = None


# ==================== 向量存储配置 ====================

@dataclass
class BatchConfig:
    """批处理配置"""
    enabled: Optional[bool] = None
    batch_size: Optional[int] = None
    show_progress: Optional[bool] = None
    max_concurrent_batches: Optional[int] = None


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    persist_directory: Optional[str] = None
    collection_prefix: Optional[str] = None
    embedding: Optional[EmbeddingConfig] = None
    batch_processing: Optional[BatchConfig] = None


# ==================== 环境变量加载工具 ====================

def load_env_overrides() -> Dict[str, Any]:
    """加载环境变量覆盖"""
    load_dotenv()
    overrides = {}
    env_map = {
        'llm_provider': 'LLM_PROVIDER',
        'llm_api_key': 'LLM_API_KEY',
        'llm_api_base': 'LLM_API_BASE',
        'llm_model_name': 'LLM_MODEL_NAME',
        'llm_reasoning_effort': 'LLM_REASONING_EFFORT',
        'embedding_model_path': 'EMBEDDING_MODEL_PATH',
        'cross_encoder_model_path': 'CROSS_ENCODER_MODEL_PATH',
    }
    for key, env_var in env_map.items():
        value = os.getenv(env_var)
        if value:
            overrides[key] = value
    return overrides


# ==================== YAML 配置加载工具 ====================

def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件，返回原始字典"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def build_config_from_dict(config_class, data: dict):
    """使用 dacite 从字典构建 dataclass 配置"""
    return from_dict(config_class, data)


# ==================== 统一配置管理器 ====================

class UnifiedConfigManager:
    """QA 系统统一配置管理器"""

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> QASystemConfig:
        """加载 QA 系统配置，合并 YAML + 环境变量"""
        try:
            config = load_yaml_config(config_path)

            qa = config.get('qa_system', {})
            vector = config.get('vector_processing', {})
            data = config.get('data_processing', {})

            # 环境变量覆盖（优先级最高）
            env = load_env_overrides()

            if 'llm' not in qa:
                qa['llm'] = {}

            llm_env_map = {
                'llm_provider': 'provider',
                'llm_api_key': 'api_key',
                'llm_api_base': 'api_base',
                'llm_model_name': 'model_name',
                'llm_reasoning_effort': 'reasoning_effort',
            }
            for env_key, cfg_key in llm_env_map.items():
                if env.get(env_key):
                    qa['llm'][cfg_key] = env[env_key]

            if env.get('embedding_model_path'):
                if 'embedding' not in vector:
                    vector['embedding'] = {}
                vector['embedding']['local_path'] = env['embedding_model_path']

            if env.get('cross_encoder_model_path'):
                if 'rerank' not in qa:
                    qa['rerank'] = {}
                qa['rerank']['cross_encoder_local_path'] = env['cross_encoder_model_path']

            config_dict = {}
            config_dict.update(qa)
            config_dict['data_dir'] = data.get("output_dir")
            config_dict['embedding'] = vector.get("embedding")

            return build_config_from_dict(QASystemConfig, config_dict)

        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            return QASystemConfig()


class VectorStoreConfigManager:
    """向量存储配置管理器"""

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> VectorStoreConfig:
        """从 YAML 加载向量存储配置"""
        config_dict = load_yaml_config(config_path)
        vector_config = config_dict.get('vector_processing', {})
        return build_config_from_dict(VectorStoreConfig, vector_config)


class DataProcessingConfigManager:
    """数据处理配置管理器"""

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Optional[ProcessingConfig]:
        """从 YAML 加载数据处理配置"""
        try:
            config_dict = load_yaml_config(config_path)
            data_config = config_dict.get('data_processing', {})
            return build_config_from_dict(ProcessingConfig, data_config)
        except FileNotFoundError:
            logging.error(f"配置文件不存在: {config_path}")
            return None
        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            return None
