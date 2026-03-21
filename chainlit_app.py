#!/usr/bin/env python3
"""
Chainlit Web 界面
支持 PostgreSQL 历史记录、文档上传、问答系统
"""

import sys
import os
import asyncio
import logging
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv
import threading

# 禁用 Chainlit 的 SQLAlchemy 相关 warning（JSONB 查询兼容性问题）
logging.getLogger('chainlit').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

try:
    from query_system import QASystem, UnifiedConfigManager
except ImportError:
    print("❌ 请确保 query_system.py 在同一目录下")
    QASystem = None
    UnifiedConfigManager = None
    sys.exit(1)

load_dotenv()

# ===================== 数据库配置 =====================

os.environ['CHAINLIT_AUTH_SECRET'] = os.getenv('CHAINLIT_AUTH_SECRET', 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6')
os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgresql+asyncpg://chainlit_user:060906@localhost:5432/chainlit_db')

# ===================== 认证配置 =====================

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    users = {
        "Syway": "060906",
    }
    if username in users and users[username] == password:
        return cl.User(identifier=username)
    return None


# ===================== 全局系统初始化 =====================

# 使用线程锁保护单例模式
_init_lock = threading.Lock()
_qa_system_instance = None


def get_qa_system():
    """
    线程安全的问答系统实例获取 - 使用双重检查锁定模式
    """
    global _qa_system_instance

    # 双重检查锁定模式
    if _qa_system_instance is None:
        with _init_lock:  # 获取锁
            if _qa_system_instance is None:  # 二次检查
                print("🚀 正在初始化问答系统...")
                try:
                    config = UnifiedConfigManager.load_config()
                    _qa_system_instance = QASystem(config)

                    # 验证系统是否正常
                    if hasattr(_qa_system_instance, 'vector_manager'):
                        store_count = len(_qa_system_instance.vector_manager.vector_stores)
                        print(f"✅ 系统初始化完成，加载向量库: {store_count} 个")

                    # 验证实例有效性
                    if _qa_system_instance is None:
                        raise RuntimeError("问答系统初始化失败，实例为空")

                except Exception as e:
                    print(f"❌ 系统初始化失败: {e}")
                    _qa_system_instance = None  # 重置实例以便后续重试
                    raise

    return _qa_system_instance


# ===================== 数据层配置（历史记录） =====================

@cl.data_layer
def get_data_layer():
    """
    建立 SQLAlchemy 数据层，用于将聊天历史持久化到 PostgreSQL
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL 环境变量未设置")
    return SQLAlchemyDataLayer(conninfo=db_url)


# ===================== Chainlit事件处理器 =====================

@cl.on_chat_start
async def on_chat_start():
    """
    新建对话时执行
    """
    try:
        # 获取问答系统实例
        qa_system = get_qa_system()

        # 将系统实例保存在用户会话中
        cl.user_session.set("qa_system", qa_system)

        # 初始化展示检索文档的开关状态（默认关闭）
        cl.user_session.set("show_retrieved_docs", False)

        # 初始化聊天历史（新建对话时清空历史）
        cl.user_session.set("chat_history", [])

        # 获取配置信息用于显示
        config = qa_system.config

        # 发送欢迎消息
        welcome_msg = f"""
# AI课程智能体

## 系统状态
- ✅ 问答系统已就绪
- 📚 已加载 {len(qa_system.vector_manager.vector_stores)} 个知识库
- 🤖 使用模型: {config.llm.model_name}
- ⚙️ 检索配置: 每个库 {config.retrieval.k_per_store} 个文档，最多 {config.retrieval.total_max_k} 个
- 📄 文档展示: **关闭**

## 控制命令
- `/show_docs` - 启用显示检索文档
- `/hide_docs` - 禁用显示检索文档
- `/status` - 查看当前开关状态

## 示例问题
- 介绍一下BERT
- 介绍一下神经网络
- 什么是Transformer？
"""

        await cl.Message(content=welcome_msg).send()

    except Exception as e:
        error_msg = f"❌ 系统启动失败: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    恢复历史会话时执行
    加载历史消息，但不发送欢迎消息
    """
    try:
        # 获取问答系统实例
        qa_system = get_qa_system()
        cl.user_session.set("qa_system", qa_system)

        # 初始化展示检索文档的开关状态
        cl.user_session.set("show_retrieved_docs", False)

        # 恢复聊天历史
        chat_history = []
        for message in thread['steps']:
            if message['type'] == 'user_message':
                chat_history.append({'role': 'user', 'content': message.get('output', '')})
            elif message['type'] == 'assistant_message':
                chat_history.append({'role': 'assistant', 'content': message.get('output', '')})

        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        error_msg = f"❌ 会话恢复失败: {str(e)}"
        await cl.Message(content=error_msg).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    处理用户消息
    核心问答逻辑，复用 QASystem.query() 方法
    同时处理控制命令和文档上传
    
    支持多轮对话上下文（仅保留问答摘要，不包含检索文档）
    """
    user_input = str(message.content).strip()
    
    if not user_input:
        await cl.Message(content="请输入有效的问题").send()
        return

    # 获取当前开关状态
    show_docs = cl.user_session.get("show_retrieved_docs")
    if show_docs is None:
        show_docs = True  # 默认启用
        cl.user_session.set("show_retrieved_docs", show_docs)

    # 检查是否为控制命令
    if user_input.lower() in ["/show_docs", "/hide_docs", "/status", "show_docs", "hide_docs", "status"]:
        await handle_control_command(user_input, show_docs)
        return

    response_msg = None

    try:
        # 从用户会话中获取问答系统
        qa_system = cl.user_session.get("qa_system")
        if not qa_system:
            # 如果会话中没有系统实例，重新获取
            qa_system = get_qa_system()
            cl.user_session.set("qa_system", qa_system)

        # 获取历史对话（用于多轮对话上下文）
        chat_history = cl.user_session.get("chat_history", [])
        
        # 如果有历史对话，将历史对话作为上下文，不包含检索到的文档
        if chat_history:
            # 构建带上下文的 prompt（不限制轮数）
            context = "\n\n".join([
                f"用户：{msg['content']}" if msg['role'] == 'user' 
                else f"助手：{msg['content']}"
                for msg in chat_history
            ])
            user_input_with_context = f"对话历史：\n{context}\n\n当前问题：{user_input}"
        else:
            user_input_with_context = user_input

        # 创建消息对象（用于流式输出）
        response_msg = cl.Message(content="")
        await response_msg.send()

        # 检查系统配置，判断是否使用流式输出
        if qa_system.config.system.streaming_output:
            # 使用流式查询
            await stream_response(qa_system, user_input_with_context, response_msg)
        else:
            # 使用普通查询
            await normal_response(qa_system, user_input_with_context, response_msg)

        # 更新历史对话（只保留问答摘要，不包含检索文档）
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_msg.content})
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        error_msg = f"❌ 查询过程中出现错误：{str(e)}"
        if response_msg:
            response_msg.content = error_msg
            await response_msg.update()
        else:
            await cl.Message(content=error_msg).send()


# ===================== 响应处理函数 =====================

async def handle_control_command(command: str, current_state: bool):
    """
    处理控制命令
    """
    command = command.lower().strip()
    
    if command in ["/show_docs", "show_docs"]:
        cl.user_session.set("show_retrieved_docs", True)
        status_msg = "## ✅ 命令执行成功\n\n📄 **检索文档展示已启用**\n\n当前状态: **启用**"
        await cl.Message(content=status_msg).send()
        
    elif command in ["/hide_docs", "hide_docs"]:
        cl.user_session.set("show_retrieved_docs", False)
        status_msg = "## ✅ 命令执行成功\n\n📄 **检索文档展示已禁用**\n\n当前状态: **禁用**"
        await cl.Message(content=status_msg).send()
        
    elif command in ["/status", "status"]:
        show_docs = cl.user_session.get("show_retrieved_docs")
        status_text = "启用" if show_docs else "禁用"
        status_msg = f"## 📊 当前状态\n\n📄 **检索文档展示**: **{status_text}**"
        await cl.Message(content=status_msg).send()


async def display_retrieved_docs(retrieved_docs: list):
    """展示检索到的文档列表"""
    if not retrieved_docs:
        return

    # 构建文档展示内容 - 使用列表推导式更高效
    docs_parts = [f"## 📚 检索到的文档\n\n共找到 **{len(retrieved_docs)}** 个相关文档片段\n\n"]

    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})

        # 使用更简洁的变量命名和默认值
        card = [
            f"### 📄 文档 {i}\n\n",
            f"**章节**: {meta.get('chapter_path', '')}\n\n",
            f"**来源**: {meta.get('file_name', meta.get('source', '未知文档'))}\n\n",
            f"**相关度**: {meta.get('similarity_score', 0):.4f}\n\n",
            f"**向量库**: {meta.get('source_store', '未知库')}\n\n",
            f"**内容**:\n```\n{doc.get('content', '')}\n```\n\n",
            "---\n\n"
        ]
        docs_parts.extend(card)

    # 一次性发送
    await cl.Message(content="".join(docs_parts)).send()

async def stream_response(qa_system, user_input: str, msg: cl.Message):
    """处理流式响应"""
    full_response = ""
    retrieved_docs = []

    try:
        if hasattr(qa_system, 'stream_query_with_docs'):
            token_stream, retrieved_docs = qa_system.stream_query_with_docs(user_input)
        else:
            token_stream = qa_system.stream_query(user_input)

        for chunk in token_stream:
            if chunk:
                await msg.stream_token(chunk)
                full_response += chunk
                await asyncio.sleep(0.01)  # 避免发送过快

        if not hasattr(qa_system, 'stream_query_with_docs'):
            result = qa_system.query(user_input, verbose=False, return_retrieved_docs=True)
            if isinstance(result, dict):
                retrieved_docs = result.get("retrieved_docs", [])

    except Exception as e:
        await msg.stream_token(f"\n\n❌ 流式输出异常: {str(e)}")

    # 更新消息完成状态
    await msg.update()

    # 根据开关状态决定是否显示检索文档
    show_docs = cl.user_session.get("show_retrieved_docs")
    if show_docs is None:
        show_docs = False
    if not show_docs:
        return

    if retrieved_docs:
        await display_retrieved_docs(retrieved_docs)


async def normal_response(qa_system, user_input: str, msg: cl.Message):
    """处理普通响应（带超时和错误处理）"""
    try:
        # 设置超时时间的查询
        def query_with_timeout():
            return qa_system.query(user_input, verbose=False, return_retrieved_docs=True)

        response = await asyncio.wait_for(
            asyncio.to_thread(query_with_timeout),
            timeout=60.0
        )

        # 解析响应
        if isinstance(response, dict):
            answer = response.get("answer", "")
            retrieved_docs = response.get("retrieved_docs", [])
        else:
            answer = response
            retrieved_docs = []

        # 显示答案
        msg.content = answer
        await msg.update()

        # 根据开关状态决定是否显示检索到的文档
        show_docs = cl.user_session.get("show_retrieved_docs")
        if show_docs is None:
            show_docs = True
        
        if show_docs and retrieved_docs:
            await display_retrieved_docs(retrieved_docs)

    except asyncio.TimeoutError:
        msg.content = "❌ 查询超时，请稍后重试"
        await msg.update()
    except Exception as e:
        msg.content = f"❌ 查询失败: {str(e)}"
        await msg.update()


# ===================== 辅助功能 =====================

@cl.action_callback("show_system_info")
async def on_action_show_info():
    """
    显示系统信息
    """
    qa_system = cl.user_session.get("qa_system")
    if not qa_system:
        await cl.Message(content="❌ 系统未就绪").send()
        return

    try:
        # 获取系统状态信息
        store_info = qa_system.vector_manager.get_store_info()

        info_text = "## 系统状态信息\n\n"
        info_text += f"- **模型**: {qa_system.config.llm.model_name}\n"
        info_text += f"- **向量库数量**: {len(store_info)}\n"

        total_docs = 0
        for store_id, info in store_info.items():
            docs = info.get('num_docs', 0)
            total_docs += docs

        info_text += f"- **总文档数**: {total_docs}\n"
        info_text += f"- **检索配置**: 每个库 {qa_system.config.retrieval.k_per_store} 个文档，最多 {qa_system.config.retrieval.total_max_k} 个\n"
        info_text += f"- **流式输出**: {'启用' if qa_system.config.system.streaming_output else '禁用'}\n"

        await cl.Message(content=info_text).send()
    except Exception as e:
        await cl.Message(content=f"❌ 获取系统信息失败: {str(e)}").send()


@cl.action_callback("list_vector_stores")
async def on_action_list_stores():
    """
    列出所有向量库
    """
    qa_system = cl.user_session.get("qa_system")
    if not qa_system:
        await cl.Message(content="❌ 系统未就绪").send()
        return

    try:
        store_info = qa_system.vector_manager.get_store_info()

        stores_text = "## 📚 向量库列表\n\n"
        if not store_info:
            stores_text = "❌ 未找到任何向量库"
        else:
            for store_id, info in store_info.items():
                stores_text += f"### {store_id}\n"
                stores_text += f"- 文档数: {info.get('num_docs', 0)}\n"
                stores_text += f"- PDF文件: {info.get('pdf_name', '未知')}\n"
                if 'original_pdf' in info:
                    stores_text += f"- 原始路径: {info.get('original_pdf')}\n"
                stores_text += "\n"

        await cl.Message(content=stores_text).send()
    except Exception as e:
        await cl.Message(content=f"❌ 获取向量库列表失败: {str(e)}").send()


# ===================== 应用配置 =====================

# Chainlit应用元数据
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="标准模式",
            markdown_description="基于知识库的标准问答模式",
        ),
    ]


# ===================== 启动信息 =====================

if __name__ == "__main__":
    print("启动命令：chainlit run chainlit_app.py -w")
    print("=" * 70)
