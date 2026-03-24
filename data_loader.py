#!/usr/bin/env python3
"""
Markdown文件处理
"""

import os
import json
import re
import logging
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import yaml
from dacite import from_dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ==================== 配置管理 ====================

@dataclass
class MarkdownChunkerConfig:
    """Markdown分块器配置"""
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
    """Markdown文本清洗配置"""
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
    """完整处理配置"""
    output_dir: Optional[str] = None
    markdown_chunker: Optional[MarkdownChunkerConfig] = None
    cleaner: Optional[CleanerConfig] = None
    streaming: Optional[StreamingConfig] = None
    skip_existing: Optional[bool] = None


class ConfigManager:
    """配置管理器"""
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Optional[ProcessingConfig]:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                data_config = config_dict.get('data_processing', {})
                config = from_dict(ProcessingConfig, data_config)

                return config
        except FileNotFoundError:
            logging.error(f"配置文件不存在: {config_path}")
            return None
        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            return None


# ==================== Markdown专用分块器 ====================

class MarkdownTextSplitter:
    """Markdown专用文本分割器 - 基于文档结构的分块"""

    def __init__(self, markdown_config: MarkdownChunkerConfig, cleaner_config: CleanerConfig):
        self.markdown_config = markdown_config
        self.cleaner_config = cleaner_config
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译所有正则表达式模式"""
        patterns = {
            # 标题模式（支持1-6级）
            'heading': re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.MULTILINE),
            # 中文数字标题模式（支持"第X章"、"X.Y"、"X.Y.Z"等格式）
            'chinese_heading': re.compile(r'^(第[0-9]+章\s+.+)$|^\d+\.\d+(?:\.\d+)*\s+.+$', re.MULTILINE),
            # 代码块模式
            'code_block': re.compile(r'^```[\s\S]*?^```', re.MULTILINE | re.DOTALL),
            # 行内代码
            'inline_code': re.compile(r'`[^`\n]+`'),
            # 段落分隔（空行）
            'paragraph': re.compile(r'\n\s*\n'),
            # Markdown元数据（YAML front matter）
            'metadata': re.compile(r'^---\s*\n[\s\S]*?\n---\s*\n', re.MULTILINE),
            # 列表项
            'list_item': re.compile(r'^[*\-+]\s+', re.MULTILINE),
            # 数字列表项
            'numbered_item': re.compile(r'^\d+\.\s+', re.MULTILINE),
        }
        return patterns

    def clean_markdown_text(self, text: str) -> str:
        """清洗Markdown文本"""
        # 提取并保护代码块
        code_blocks = []
        placeholder_template = "__CODE_BLOCK_{}__"

        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return placeholder_template.format(len(code_blocks) - 1)
        
        # 使用代码块模式保护代码内容
        text = self._compiled_patterns['code_block'].sub(replace_code_block, text)
        
        # 应用清洗配置
        if self.cleaner_config.remove_metadata:
            text = self._compiled_patterns['metadata'].sub('', text)

        if self.cleaner_config.remove_empty_lines:
            # 避免影响代码块内的空行
            text = re.sub(r'\n{3,}', '\n\n', text)

        if self.cleaner_config.normalize_whitespace:
            # 只对非代码块的文本进行空格规范化
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'[ \t]+\n', '\n', text)

        # 恢复代码块
        for i, code_block in enumerate(code_blocks):
            placeholder = placeholder_template.format(i)
            text = text.replace(placeholder, code_block)

        return text.strip()

    def _extract_headings_and_content(self, text: str) -> List[Dict[str, Any]]:
        """提取标题和内容结构"""
        lines = text.split('\n')
        sections = []
        current_section = {
            'heading': None,
            'heading_level': 0,
            'content': [],
            'type': 'content',  # 'heading' 或 'content'
            'chapter_path': None  # 完整章节路径
        }

        in_code_block = False
        code_block_content = []
        heading_stack = []  # 追踪所有层级的标题

        for line in lines:
            line = line.rstrip()

            # 处理代码块开始/结束
            if line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_content = [line]
                else:
                    in_code_block = False
                    code_block_content.append(line)
                    code_text = '\n'.join(code_block_content)
                    if self.markdown_config.include_code_blocks and len(
                            code_block_content) >= self.markdown_config.code_block_min_lines:
                        current_section['content'].append(code_text)
                    code_block_content = []
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            # 标题检测：直接基于#号数量判断级别
            heading_level = 0
            heading_text = ''
            
            if line.startswith('#'):
                # 计算#号数量
                hash_count = len(line) - len(line.lstrip('#'))
                if 1 <= hash_count <= 6:
                    heading_level = hash_count
                    heading_text = line.lstrip('#').strip()

            # 如果是标题且级别在配置范围内
            if heading_level > 0 and heading_level <= self.markdown_config.heading_level:
                # 保存当前section
                if current_section['content'] or current_section['heading']:
                    sections.append(current_section.copy())

                # 特殊处理：如果是"目录"，不加入标题栈
                if heading_text.strip() == "目录":
                    heading_stack = []
                    chapter_path = heading_text
                else:
                    # 根据标题级别更新 heading_stack
                    if heading_level == 1:
                        heading_stack = [heading_text]
                    elif heading_level == 2:
                        heading_stack = heading_stack[:1] + [heading_text]
                    elif heading_level == 3:
                        heading_stack = heading_stack[:2] + [heading_text]
                    elif heading_level == 4:
                        heading_stack = heading_stack[:3] + [heading_text]
                    else:
                        # 5级及以上的标题，只保留前4级
                        heading_stack = heading_stack[:4] + [heading_text]

                    # 构建章节路径
                    chapter_path = ' -> '.join(heading_stack) if heading_stack else heading_text

                # 开始新的section
                current_section = {
                    'heading': heading_text,
                    'heading_level': heading_level,
                    'content': [],
                    'type': 'heading',
                    'chapter_path': chapter_path  # 记录完整章节路径
                }
            else:
                current_section['content'].append(line)

        # 添加最后一个section
        if current_section['content'] or current_section['heading']:
            sections.append(current_section)

        return sections

    def _split_section_into_chunks(self, section: Dict[str, Any]) -> List[str]:
        """将单个section分割成合适的chunks"""
        content_text = '\n'.join(section['content'])
        if not content_text.strip():
            return []

        chunks = []

        # 如果section很小，直接作为一个chunk
        if len(content_text) <= self.markdown_config.max_chunk_chars:
            chunk_text = ''
            if section['type'] == 'heading' and self.markdown_config.preserve_headings:
                heading_mark = '#' * section['heading_level']
                chunk_text = f"{heading_mark} {section['heading']}\n\n"
            chunk_text += content_text
            chunks.append(chunk_text.strip())
            return chunks

        # 按段落分割
        paragraphs = self._compiled_patterns['paragraph'].split(content_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            para_length = len(paragraph)

            # 如果段落本身超过最大chunk大小，需要进一步分割
            if para_length > self.markdown_config.max_chunk_chars:
                # 先保存当前chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if section['type'] == 'heading' and self.markdown_config.preserve_headings and not any(
                            '#' in line for line in current_chunk):
                        heading_mark = '#' * section['heading_level']
                        chunk_text = f"{heading_mark} {section['heading']}\n\n{chunk_text}"
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0

                # 分割大段落
                sub_chunks = self._split_large_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) >= self.markdown_config.min_chunk_chars:
                        chunks.append(sub_chunk)
                    elif self.markdown_config.combine_small_paragraphs and current_chunk:
                        # 尝试合并到前一个chunk
                        if current_length + len(sub_chunk) <= self.markdown_config.max_chunk_chars:
                            current_chunk.append(sub_chunk)
                            current_length += len(sub_chunk)
                        else:
                            chunk_text = '\n\n'.join(current_chunk)
                            if section['type'] == 'heading' and self.markdown_config.preserve_headings and not any(
                                    '#' in line for line in current_chunk):
                                heading_mark = '#' * section['heading_level']
                                chunk_text = f"{heading_mark} {section['heading']}\n\n{chunk_text}"
                            chunks.append(chunk_text)
                            current_chunk = [sub_chunk]
                            current_length = len(sub_chunk)
                    else:
                        # 单独的小段落
                        chunks.append(sub_chunk)
            else:
                # 如果当前chunk加上这个段落不会超过限制，就合并
                if current_length + para_length + 2 <= self.markdown_config.max_chunk_chars:  # +2 for \n\n
                    current_chunk.append(paragraph)
                    current_length += para_length + 2
                else:
                    # 保存当前chunk并开始新的
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        if section['type'] == 'heading' and self.markdown_config.preserve_headings and not any(
                                '#' in line for line in current_chunk):
                            heading_mark = '#' * section['heading_level']
                            chunk_text = f"{heading_mark} {section['heading']}\n\n{chunk_text}"
                        chunks.append(chunk_text)

                    # 如果段落本身足够大，单独成chunk
                    if para_length >= self.markdown_config.min_chunk_chars:
                        current_chunk = [paragraph]
                        current_length = para_length
                    else:
                        # 小段落，开始积累
                        current_chunk = [paragraph]
                        current_length = para_length

        # 处理最后一个chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if section['type'] == 'heading' and self.markdown_config.preserve_headings and not any(
                    '#' in line for line in current_chunk):
                heading_mark = '#' * section['heading_level']
                chunk_text = f"{heading_mark} {section['heading']}\n\n{chunk_text}"
            chunks.append(chunk_text)

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """分割大段落（超过最大chunk大小）"""
        # 按句子分割（中文句号、感叹号、问号）
        sentences = re.split(r'([。！？；.!?;]+\s*)', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 重组句子为chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence)
            if current_length + sent_length <= self.markdown_config.max_chunk_chars:
                current_chunk.append(sentence)
                current_length += sent_length
            else:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_length

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档（主入口）"""
        all_splits = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata

            # 清洗文本
            cleaned_text = self.clean_markdown_text(text)

            # 提取标题和内容结构
            sections = self._extract_headings_and_content(cleaned_text)

            # 分割每个section
            for section in sections:
                chunks = self._split_section_into_chunks(section)
                for chunk in chunks:
                    # 添加chunk类型到元数据
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_type': section['type'],
                        'chunking_method': 'markdown_structured',
                        'has_heading': section['type'] == 'heading',
                        'heading_level': section['heading_level'] if section['type'] == 'heading' else 0,
                        'heading_text': section['heading'] if section['type'] == 'heading' else '',
                        'chapter_path': section.get('chapter_path')  # 添加完整章节路径
                    })

                    split_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    all_splits.append(split_doc)

        logger.info(f"Markdown分块完成: {len(documents)} 个文档 -> {len(all_splits)} 个文本块")
        return all_splits

    def split_text(self, text: str) -> List[str]:
        """分割文本（简化接口）"""
        doc = Document(page_content=text, metadata={})
        split_docs = self.split_documents([doc])
        return [doc.page_content for doc in split_docs]


# ==================== Markdown加载器 ====================

class MarkdownLoader:
    """Markdown文件加载器"""

    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """加载Markdown文件"""
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                content = f.read()

            # 提取文件名和基本信息
            file_name = os.path.basename(self.file_path)
            file_size = os.path.getsize(self.file_path)

            doc = Document(
                page_content=content,
                metadata={
                    'source': self.file_path,
                    'file_name': file_name,
                    'file_size': file_size,
                    'file_type': 'markdown',
                    'encoding': self.encoding
                }
            )

            return [doc]
        except UnicodeDecodeError:
            # 尝试其他编码
            encodings = ['utf-8-sig', 'gbk', 'gb2312', 'latin-1']
            for enc in encodings:
                try:
                    with open(self.file_path, 'r', encoding=enc) as f:
                        content = f.read()

                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': self.file_path,
                            'file_name': os.path.basename(self.file_path),
                            'file_size': os.path.getsize(self.file_path),
                            'file_type': 'markdown',
                            'encoding': enc
                        }
                    )
                    return [doc]
                except:
                    continue
            raise ValueError(f"无法读取文件编码: {self.file_path}")
        except Exception as e:
            logger.error(f"加载Markdown文件失败 {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """流式加载（为兼容性保留）"""
        yield self.load()


# ==================== 数据处理核心类 ====================

class DataProcessor:
    """Markdown数据处理类"""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ConfigManager.load_config()
        self.markdown_splitter = MarkdownTextSplitter(
            self.config.markdown_chunker,
            self.config.cleaner
        )
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("DataProcessor初始化完成 (Markdown专用版)")

    def process_single_file(self, file_path: str, output_dir: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
        """处理单个Markdown文件"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 生成数据ID
        file_mtime = os.path.getmtime(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 添加父目录信息避免同名冲突
        parent_dir = os.path.basename(os.path.dirname(file_path))
        if parent_dir and parent_dir not in ['.', '..']:
            safe_parent = parent_dir.replace(' ', '_').replace('\\', '_').replace('/', '_')
            data_id = f"{safe_parent}_{file_name}_{int(file_mtime)}"
        else:
            data_id = f"{file_name}_{int(file_mtime)}"

        output_file = os.path.join(output_dir, f"{data_id}.json")
        meta_file = os.path.join(output_dir, f"{data_id}_meta.json")

        # 检查是否跳过
        if self.config.skip_existing and os.path.exists(output_file):
            logger.info(f"发现现有处理结果，跳过: {os.path.basename(file_path)}")
            return output_file, data_id, file_path

        logger.info(f"开始处理: {os.path.basename(file_path)}")

        try:
            # 加载文件
            loader = MarkdownLoader(file_path)
            documents = loader.load()

            # 分块处理
            chunks = self.markdown_splitter.split_documents(documents)

            # 序列化
            serialized_chunks = []
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                metadata.update({
                    'original_file': file_path,
                    'file_name': os.path.basename(file_path),
                    'data_id': data_id,
                    'processing_time': datetime.now().isoformat()
                })

                serialized_chunks.append({
                    'page_content': chunk.page_content,
                    'metadata': metadata
                })

            # 保存数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_chunks, f, ensure_ascii=False, indent=2)

            # 计算统计信息
            chunk_lengths = [len(chunk["page_content"]) for chunk in serialized_chunks]
            avg_chunk_size = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

            # 构建元数据
            meta_info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_mtime': file_mtime,
                'processed_time': datetime.now().isoformat(),
                'total_chunks': len(serialized_chunks),
                'data_id': data_id,
                'output_file': output_file,
                'avg_chunk_size': avg_chunk_size,
                'min_chunk_size': min(chunk_lengths) if chunk_lengths else 0,
                'max_chunk_size': max(chunk_lengths) if chunk_lengths else 0,
                'chunking_method': 'markdown_structured',
                'config': {
                    'max_chunk_chars': self.config.markdown_chunker.max_chunk_chars,
                    'min_chunk_chars': self.config.markdown_chunker.min_chunk_chars,
                    'heading_level': self.config.markdown_chunker.heading_level
                }
            }

            # 保存元数据
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 处理完成: {os.path.basename(file_path)}")
            logger.info(f"   生成 {len(serialized_chunks)} 个文本块")
            logger.info(f"   平均块大小: {avg_chunk_size:.1f} 字符")
            logger.info(f"   保存到: {output_file}")

            return output_file, data_id, file_path

        except Exception as e:
            logger.error(f"处理失败 {file_path}: {e}")
            raise

    def process_and_save(self, input_path: str, output_dir: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """处理单个文件或整个文件夹"""
        output_dir = output_dir or self.config.output_dir

        logger.info("=" * 60)
        logger.info(f"开始处理: {input_path}")
        logger.info(f"Markdown分块配置:")
        logger.info(f"  - 最大块大小: {self.config.markdown_chunker.max_chunk_chars} 字符")
        logger.info(f"  - 最小块大小: {self.config.markdown_chunker.min_chunk_chars} 字符")
        logger.info(f"  - 标题级别: {self.config.markdown_chunker.heading_level}")
        logger.info("=" * 60)

        results = []

        try:
            if os.path.isfile(input_path):
                if input_path.lower().endswith(('.md', '.markdown')):
                    result = self.process_single_file(input_path, output_dir)
                    if result:
                        results.append(result)
                else:
                    logger.warning(f"跳过非Markdown文件: {input_path}")

            elif os.path.isdir(input_path):
                # 收集所有Markdown文件
                md_files = []
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(('.md', '.markdown')):
                            md_files.append(os.path.join(root, file))

                if not md_files:
                    logger.warning(f"文件夹中没有找到Markdown文件: {input_path}")
                    return results

                logger.info(f"找到 {len(md_files)} 个Markdown文件")

                # 处理每个文件
                for i, md_file in enumerate(md_files, 1):
                    logger.info(f"\n处理文件 {i}/{len(md_files)}: {os.path.basename(md_file)}")
                    try:
                        result = self.process_single_file(md_file, output_dir)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"处理失败 {md_file}: {e}")
                        continue
            else:
                raise FileNotFoundError(f"路径不存在: {input_path}")

        except Exception as e:
            logger.error(f"处理过程发生错误: {e}")
            raise

        # 生成汇总报告
        if results:
            self._generate_summary_report(results, output_dir)

        return results

    def _generate_summary_report(self, results: List[Tuple[str, str, str]], output_dir: str):
        """生成处理汇总报告"""
        summary_file = os.path.join(output_dir, "processing_summary.json")

        summary = {
            "processed_at": datetime.now().isoformat(),
            "total_files": len(results),
            "total_chunks": 0,
            "files": [],
            "config": {
                "markdown_chunker": asdict(self.config.markdown_chunker),
                "cleaner": asdict(self.config.cleaner)
            }
        }

        for data_file, data_id, file_path in results:
            meta_file = os.path.join(output_dir, f"{data_id}_meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)

                    total_chunks = meta.get("total_chunks", 0)
                    summary["total_chunks"] += total_chunks

                    summary["files"].append({
                        "file_name": meta.get("file_name"),
                        "file_path": meta.get("file_path"),
                        "chunks": total_chunks,
                        "avg_chunk_size": meta.get("avg_chunk_size", 0),
                        "data_file": data_file,
                        "data_id": data_id
                    })
                except Exception as e:
                    logger.error(f"读取元数据文件失败 {meta_file}: {e}")
                    continue

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("📊 Markdown批量处理汇总")
        logger.info("=" * 60)
        logger.info(f"✅ 成功处理文件数: {len(results)}")
        logger.info(f"📄 总文本块数: {summary['total_chunks']}")
        logger.info(f"📝 汇总报告: {summary_file}")

    def get_processed_files_info(self, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取已处理文件的信息"""
        output_dir = output_dir or self.config.output_dir

        if not os.path.exists(output_dir):
            return []

        info_list = []

        # 读取汇总报告（如果存在）
        summary_file = os.path.join(output_dir, "processing_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                return summary.get("files", [])

        # 扫描所有元数据文件
        for file in os.listdir(output_dir):
            if file.endswith('_meta.json'):
                meta_file = os.path.join(output_dir, file)
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        info_list.append({
                            "file_name": meta.get("file_name"),
                            "data_id": meta.get("data_id"),
                            "chunks": meta.get("total_chunks", 0),
                            "avg_chunk_size": meta.get("avg_chunk_size", 0),
                            "processed_time": meta.get("processed_time"),
                            "data_file": meta.get("output_file"),
                            "chunking_method": meta.get("chunking_method", "markdown_structured")
                        })
                except:
                    continue

        return info_list

    def get_data_file_by_id(self, data_id: str, output_dir: Optional[str] = None) -> Optional[str]:
        """根据data_id获取对应的数据文件路径"""
        output_dir = output_dir or self.config.output_dir
        data_file = os.path.join(output_dir, f"{data_id}.json")

        if os.path.exists(data_file):
            return data_file
        return None


# ==================== 配置常量 ====================
# 将硬编码值提取为常量，便于统一修改
DEFAULT_OUTPUT_DIR = "./processed_data"
DEFAULT_LOG_FILE = "logs/data_processing.log"
MAX_PREVIEW_LENGTH = 100
DISPLAY_WIDTH = 60
SEPARATOR = "=" * DISPLAY_WIDTH
SHORT_SEPARATOR = "-" * DISPLAY_WIDTH

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# ==================== 日志配置 ====================

# 确保日志目录存在
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DEFAULT_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler() if sys.stdout.isatty() else logging.NullHandler()
    ]
)

# 模块级日志记录器
logger = logging.getLogger(__name__)


# ==================== 命令行接口 ====================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Markdown批量处理工具 - 结构化分块版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单个文件: python data_loader.py process "document.md"
  整个文件夹: python data_loader.py process "./markdown_files"
  指定输出: python data_loader.py process "document.md" --output "./my_data"
  查看列表: python data_loader.py list --output "./my_data"
  查看详情: python data_loader.py info "doc_1234567890" --output "./my_data"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    process_parser = subparsers.add_parser("process", help="处理Markdown文件或文件夹")
    process_parser.add_argument("input_path", help="Markdown文件路径或文件夹路径")
    process_parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                                help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    process_parser.add_argument("--log-level", default="INFO",
                                choices=list(LOG_LEVELS.keys()), help="日志级别")

    list_parser = subparsers.add_parser("list", help="查看已处理文件列表")
    list_parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                             help=f"数据目录 (默认: {DEFAULT_OUTPUT_DIR})")
    list_parser.add_argument("--log-level", default="INFO",
                             choices=list(LOG_LEVELS.keys()), help="日志级别")

    info_parser = subparsers.add_parser("info", help="查看指定数据ID的详细信息")
    info_parser.add_argument("data_id", help="数据ID")
    info_parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                             help=f"数据目录 (默认: {DEFAULT_OUTPUT_DIR})")
    info_parser.add_argument("--log-level", default="INFO",
                             choices=list(LOG_LEVELS.keys()), help="日志级别")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()

# ==================== 辅助函数 ====================

def _format_chunk_preview(content: str, max_length: int = MAX_PREVIEW_LENGTH) -> str:
    """格式化文本块预览"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def _load_metadata_file(data_id: str, output_dir: str) -> Optional[dict]:
    """加载元数据文件"""
    meta_file = os.path.join(output_dir, f"{data_id}_meta.json")
    if not os.path.exists(meta_file):
        return None

    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取元数据文件失败 {meta_file}: {e}")
        return None


def _load_data_file(data_id: str, output_dir: str) -> Optional[list]:
    """加载数据文件"""
    data_file = os.path.join(output_dir, f"{data_id}.json")
    if not os.path.exists(data_file):
        return None

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取数据文件失败 {data_file}: {e}")
        return None

# ==================== 辅助函数 ====================

def process_files(processor, input_path: str, output_dir: str) -> int:
    """处理文件或文件夹"""
    if os.path.isfile(input_path):
        return process_single_file(processor, input_path, output_dir)
    elif os.path.isdir(input_path):
        return process_directory(processor, input_path, output_dir)
    else:
        print(f"❌ 错误: 路径不存在: {input_path}")
        return 1


def process_single_file(processor, input_path: str, output_dir: str) -> int:
    """处理单个文件"""
    if not input_path.lower().endswith(('.md', '.markdown')):
        print(f"❌ 错误: 不是Markdown文件: {input_path}")
        return 1

    print(SEPARATOR)
    print(f"🚀 开始处理: {input_path}")
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"🧩 分块方法: Markdown结构化分块")
    print(f"⚙️  配置: 最大{processor.config.markdown_chunker.max_chunk_chars}字符, "
          f"最小{processor.config.markdown_chunker.min_chunk_chars}字符")
    print(SEPARATOR)

    try:
        results = processor.process_and_save(input_path, output_dir)

        if not results:
            print("❌ 处理失败，未生成结果")
            return 1

        data_file, data_id, file_path = results[0]

        print("\n" + SEPARATOR)
        print("🎉 单个文件处理完成！")
        print(SEPARATOR)
        print(f"📄 原文件: {file_path}")
        print(f"🔑 数据ID: {data_id}")
        print(f"💾 数据文件: {os.path.abspath(data_file)}")

        meta = _load_metadata_file(data_id, output_dir)
        if meta:
            print(f"📊 生成文本块: {meta.get('total_chunks', 0)} 个")
            print(f"📅 处理时间: {meta.get('processed_time', '未知')}")
            print(f"📏 平均块大小: {meta.get('avg_chunk_size', 0):.1f} 字符")

        if os.path.exists(data_file):
            size_kb = os.path.getsize(data_file) / 1024
            print(f"📦 数据文件大小: {size_kb:.1f} KB")

        print("\n📌 后续使用:")
        print(f"  1. 查看文件信息: python data_loader.py info {data_id}")
        print(f"  2. 查看文件列表: python data_loader.py list")

        return 0

    except Exception as e:
        logger.error(f"处理文件失败 {input_path}: {e}")
        return 1


def process_directory(processor, input_path: str, output_dir: str) -> int:
    """处理目录"""
    print(SEPARATOR)
    print(f"🚀 开始批量处理: {input_path}")
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"🧩 分块方法: Markdown结构化分块")
    print(SEPARATOR)

    try:
        results = processor.process_and_save(input_path, output_dir)

        print("\n" + SEPARATOR)
        print("🎉 批量处理完成！")
        print(SEPARATOR)

        if not results:
            print("❌ 没有找到可处理的Markdown文件")
            return 1

        print(f"✅ 成功处理文件数: {len(results)}")

        print("\n📋 处理结果摘要:")
        print(SHORT_SEPARATOR)

        total_chunks = 0
        for i, (data_file, data_id, file_path) in enumerate(results, 1):
            meta = _load_metadata_file(data_id, output_dir)

            chunks = 0
            avg_size = 0

            if meta:
                chunks = meta.get("total_chunks", 0)
                avg_size = meta.get("avg_chunk_size", 0)
                total_chunks += chunks

            filename = os.path.basename(file_path)
            print(f"{i:2d}. {filename:40s}")
            print(f"    数据ID: {data_id}")
            print(f"    文本块: {chunks:4d} 个")
            print(f"    平均大小: {avg_size:.0f} 字符")
            print(f"    数据文件: {os.path.basename(data_file)}")
            print()

        print(SHORT_SEPARATOR)
        print(f"📊 总计: {len(results)} 个文件, {total_chunks} 个文本块")
        return 0

    except Exception as e:
        logger.error(f"处理目录失败 {input_path}: {e}")
        return 1


def show_file_list(processor, output_dir: str) -> int:
    """显示已处理文件列表"""
    try:
        print("📋 已处理文件列表:")
        print(SHORT_SEPARATOR)

        info_list = processor.get_processed_files_info(output_dir)

        if not info_list:
            print("❌ 没有找到已处理的数据文件")
            print(f"请检查目录是否存在: {output_dir}")
            return 1

        print(f"找到 {len(info_list)} 个已处理文件:")
        print(SHORT_SEPARATOR)

        total_chunks = 0
        for i, info in enumerate(info_list, 1):
            chunks = info.get("chunks", 0)
            total_chunks += chunks

            method = info.get("chunking_method", "markdown_structured")
            method_display = method[:3].upper() if len(method) > 3 else method.upper()

            print(f"{i:2d}. {info.get('file_name', '未知'):40s}")
            print(f"    数据ID: {info.get('data_id', '未知')}")
            print(f"    文本块: {chunks:4d} 个 [方法: {method_display}]")
            print(f"    平均大小: {info.get('avg_chunk_size', 0):.0f} 字符")
            print(f"    处理时间: {info.get('processed_time', '未知')[:19]}")
            print(f"    数据文件: {os.path.basename(info.get('data_file', ''))}")
            print()

        print(SHORT_SEPARATOR)
        print(f"📊 总计: {len(info_list)} 个文件, {total_chunks} 个文本块")
        return 0

    except Exception as e:
        logger.error(f"显示文件列表失败: {e}")
        return 1


def show_file_info(processor, data_id: str, output_dir: str) -> int:
    """显示文件详细信息"""
    if not data_id:
        print("❌ 错误: 请提供数据ID")
        print("用法: python data_loader.py info <数据ID>")
        return 1

    print(f"🔍 数据ID: {data_id}")

    data_file = processor.get_data_file_by_id(data_id, output_dir)
    if not data_file:
        print(f"❌ 未找到数据ID: {data_id}")
        return 1

    print(f"📄 数据文件: {os.path.abspath(data_file)}")

    data = _load_data_file(data_id, output_dir)
    if not data:
        print("❌ 无法读取数据文件")
        return 1

    print(f"📊 包含 {len(data)} 个文本块")

    print("\n📝 文本块预览:")
    print(SHORT_SEPARATOR)
    for i, item in enumerate(data[:3], 1):
        content = item.get("page_content", "")
        metadata = item.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "未知")
        has_heading = metadata.get("has_heading", False)

        type_display = f"标题块" if has_heading else f"正文块"

        print(f"【块 {i} | 类型: {type_display}】")
        print(f"  长度: {len(content)} 字符")
        print(f"  原始文件: {metadata.get('original_file', '未知')}")

        # 格式化预览
        if len(content) <= MAX_PREVIEW_LENGTH:
            preview = content
        else:
            preview = content[:MAX_PREVIEW_LENGTH] + "..."
        print(f"  预览: {preview}")
        print()

    meta = _load_metadata_file(data_id, output_dir)
    if meta:
        print("📋 元数据信息:")
        print(f"  文件路径: {meta.get('file_path', '未知')}")
        print(f"  文件大小: {meta.get('file_size', 0) / 1024 / 1024:.2f} MB")
        print(f"  分块方法: {meta.get('chunking_method', '未知')}")
        print(f"  处理时间: {meta.get('processed_time', '未知')}")
        print(f"  平均块大小: {meta.get('avg_chunk_size', 0):.1f} 字符")
        print(f"  最小块大小: {meta.get('min_chunk_size', 0):.0f} 字符")
        print(f"  最大块大小: {meta.get('max_chunk_size', 0):.0f} 字符")

    return 0


# ==================== 主程序 ====================

def main() -> int:
    """主程序入口"""
    try:
        args = parse_arguments()

        processor = DataProcessor()
        output_dir = args.output

        if args.command == "process":
            # 直接在这里定义process_files函数，避免导入问题
            return process_files(processor, args.input_path, output_dir)
        elif args.command == "list":
            return show_file_list(processor, output_dir)
        elif args.command == "info":
            return show_file_info(processor, args.data_id, output_dir)
        else:
            print(f"❌ 未知命令: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 130
    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)