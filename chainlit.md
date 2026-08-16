## ReTA — AI 课程智能问答系统

基于检索增强生成（RAG）的 AI 课程智能问答系统，使用 LangChain 框架和 Chainlit 界面。

项目地址: https://github.com/Syway96/ReTA

### 控制命令

默认模型：deepseek-v4-flash（默认支持 DeepSeek API）
默认思考强度：off（关闭思考）
以下命令仅本次运行期间生效，如需持久化默认配置请修改 .env

| 命令 | 说明 |
|---|---|
| `/status` | 查看当前模型、思考模式与检索配置 |
| `/show_docs` / `/hide_docs` | 切换检索文档展示 |
| `/think off\|low\|medium\|high` | 切换思考强度（立即生效） |
| `/model <模型名>` | 切换模型（立即生效，如 `deepseek-v4-pro`、`deepseek-v4-flash`） |

### 示例问题

- 介绍一下 BERT
- 介绍一下神经网络
- 什么是 Transformer？
