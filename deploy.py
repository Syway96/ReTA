#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键部署脚本 - AI 课程智能问答系统
功能：环境检查、配置验证、向量库构建、应用启动
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Logger:
    """日志管理器"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def _log(self, level: str, message: str, color: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        if color:
            print(f"{color}{log_entry}{Colors.ENDC}")
        else:
            print(log_entry)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
    
    def info(self, message: str):
        self._log("INFO", message, Colors.OKCYAN)
    
    def success(self, message: str):
        self._log("SUCCESS", message, Colors.OKGREEN)
    
    def warning(self, message: str):
        self._log("WARNING", message, Colors.WARNING)
    
    def error(self, message: str):
        self._log("ERROR", message, Colors.FAIL)
    
    def header(self, message: str):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{message.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


class DeploymentChecker:
    """部署检查器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.project_root = Path(__file__).parent.absolute()
        self.conda_env_name = "ReTA"
        self.python_version_required = (3, 11)
        self.python_version_max = (3, 12)  # 不允许超过 3.12
        
    def check_python_version(self) -> bool:
        """检查 Python 版本（严格要求 3.11.x）"""
        current_version = sys.version_info
        required = f"{self.python_version_required[0]}.{self.python_version_required[1]}"
        
        # 严格检查：必须是 3.11.x
        if current_version.major != self.python_version_required[0] or \
           current_version.minor != self.python_version_required[1]:
            self.logger.error(f"Python 版本不符合要求：{current_version.major}.{current_version.minor}.{current_version.micro}")
            self.logger.error(f"必须使用 Python {required}.x（当前版本：{current_version.major}.{current_version.minor}）")
            self.logger.error("请创建 Python 3.11 的 Conda 环境：")
            self.logger.error("  conda create -n ReTA python=3.11")
            self.logger.error("  conda activate ReTA")
            return False
        
        self.logger.success(f"Python 版本：{current_version.major}.{current_version.minor}.{current_version.micro} ✓")
        return True
    
    def check_conda_installed(self) -> bool:
        """检查 Conda 是否安装"""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.success(f"Conda 已安装：{result.stdout.strip()} ✓")
                return True
            else:
                self.logger.error("Conda 未正确安装")
                return False
        except FileNotFoundError:
            self.logger.error("未找到 Conda 命令，请先安装 Anaconda 或 Miniconda")
            return False
        except Exception as e:
            self.logger.error(f"检查 Conda 时出错：{str(e)}")
            return False
    
    def check_conda_env_exists(self) -> bool:
        """检查 Conda 虚拟环境是否存在"""
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if self.conda_env_name in result.stdout:
                self.logger.success(f"Conda 环境 '{self.conda_env_name}' 已存在 ✓")
                return True
            else:
                self.logger.warning(f"Conda 环境 '{self.conda_env_name}' 不存在")
                return False
        except Exception as e:
            self.logger.error(f"检查 Conda 环境时出错：{str(e)}")
            return False
    
    def check_dependencies(self) -> bool:
        """检查并安装 requirements.txt 中的所有依赖"""
        requirements_path = self.project_root / "requirements.txt"
        
        if not requirements_path.exists():
            self.logger.error("requirements.txt 文件不存在")
            return False
        
        self.logger.info("正在安装 requirements.txt 中的依赖...")
        print()
        
        try:
            # 直接使用 pip install -r requirements.txt
            # 不捕获输出，让 pip 实时显示进度
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                capture_output=False,  # 不捕获输出，直接显示
                text=True,
                encoding='utf-8'
            )
            
            print()
            
            if result.returncode != 0:
                self.logger.error("安装依赖失败")
                return False
            
            self.logger.success("\n所有依赖包已安装/更新 ✓")
            return True
            
        except Exception as e:
            self.logger.error(f"安装依赖时出错：{str(e)}")
            return False
    
    def check_config_file(self) -> bool:
        """检查配置文件是否存在"""
        config_path = self.project_root / "config.yaml"
        
        if config_path.exists():
            self.logger.success("配置文件 config.yaml 存在 ✓")
            return True
        else:
            self.logger.error("配置文件 config.yaml 不存在")
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """运行所有检查"""
        results = {}
        
        self.logger.header("环境检查")
        
        results["python_version"] = self.check_python_version()
        results["conda_installed"] = self.check_conda_installed()
        
        if results["conda_installed"]:
            results["conda_env_exists"] = self.check_conda_env_exists()
            results["dependencies"] = self.check_dependencies()
        
        results["config_file"] = self.check_config_file()
        
        return results


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.project_root = Path(__file__).parent.absolute()
        self.config_path = self.project_root / "config.yaml"
        self.config = None
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.logger.success("配置文件加载成功 ✓")
            return True
        except Exception as e:
            self.logger.error(f"加载配置文件失败：{str(e)}")
            return False
    
    def validate_llm_config(self) -> bool:
        """验证 LLM 配置"""
        if not self.config:
            return False
        
        llm_config = self.config.get('qa_system', {}).get('llm', {})
        provider = llm_config.get('provider', 'api')
        
        if provider == 'api':
            api_key = llm_config.get('api_key', '')
            if not api_key or api_key.startswith('sk-') is False:
                self.logger.warning("API Key 可能未正确配置")
                return False
            else:
                self.logger.success("API Key 配置正确 ✓")
        elif provider == 'local':
            self.logger.info("使用本地 Ollama 模型")
        
        return True
    
    def update_provider(self, provider: str) -> bool:
        """更新配置文件中的 LLM Provider"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'qa_system' not in config:
                config['qa_system'] = {}
            if 'llm' not in config['qa_system']:
                config['qa_system']['llm'] = {}
            
            config['qa_system']['llm']['provider'] = provider
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.config = config
            self.logger.success(f"LLM Provider 已更新为：{provider} ✓")
            return True
        except Exception as e:
            self.logger.error(f"更新 Provider 失败：{str(e)}")
            return False
    
    def update_api_key(self, api_key: str) -> bool:
        """更新配置文件中的 API Key"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'qa_system' not in config:
                config['qa_system'] = {}
            if 'llm' not in config['qa_system']:
                config['qa_system']['llm'] = {}
            
            config['qa_system']['llm']['api_key'] = api_key
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.config = config
            self.logger.success("API Key 已更新到配置文件 ✓")
            return True
        except Exception as e:
            self.logger.error(f"更新 API Key 失败：{str(e)}")
            return False
    
    def update_api_model(self, model_name: str) -> bool:
        """更新配置文件中的 API 模型名称"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'qa_system' not in config:
                config['qa_system'] = {}
            if 'llm' not in config['qa_system']:
                config['qa_system']['llm'] = {}
            
            config['qa_system']['llm']['model_name'] = model_name
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.config = config
            self.logger.success(f"API 模型已更新为：{model_name} ✓")
            return True
        except Exception as e:
            self.logger.error(f"更新 API 模型失败：{str(e)}")
            return False
    
    def update_local_model(self, model_name: str) -> bool:
        """更新配置文件中的本地模型名称"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'qa_system' not in config:
                config['qa_system'] = {}
            if 'llm' not in config['qa_system']:
                config['qa_system']['llm'] = {}
            
            config['qa_system']['llm']['model_name'] = model_name
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.config = config
            self.logger.success(f"本地模型已更新为：{model_name} ✓")
            return True
        except Exception as e:
            self.logger.error(f"更新本地模型失败：{str(e)}")
            return False
    
    def run_validation(self) -> Dict[str, bool]:
        """运行所有验证"""
        results = {}
        
        self.logger.header("配置验证")
        
        if not self.load_config():
            return {"config_loaded": False}
        
        results["config_loaded"] = True
        results["llm_config"] = self.validate_llm_config()
        
        return results


class ModelDownloader:
    """模型下载管理器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.project_root = Path(__file__).parent.absolute()
        self.models = {
            'embedding': {
                'model_id': 'BAAI/bge-small-zh-v1.5',
                'config_paths': [
                    'vector_processing.embedding.local_path',
                    'qa_system.embedding.local_path'
                ]
            },
            'reranker': {
                'model_id': 'BAAI/bge-reranker-base',
                'config_paths': [
                    'qa_system.rerank.cross_encoder_local_path'
                ]
            }
        }
    
    def get_model_path_from_config(self, config_path: str) -> Optional[str]:
        """从配置中获取模型路径"""
        if not self.validator or not self.validator.config:
            return None
        
        keys = config_path.split('.')
        value = self.validator.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value if isinstance(value, str) else None
    
    def check_modelscope_installed(self) -> bool:
        """检查 ModelScope 是否安装"""
        try:
            import modelscope
            self.logger.success("ModelScope 已安装 ✓")
            return True
        except ImportError:
            self.logger.warning("ModelScope 未安装")
            self.logger.info("正在安装 ModelScope...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "modelscope"],
                    capture_output=False,
                    check=True
                )
                self.logger.success("ModelScope 安装成功 ✓")
                return True
            except Exception as e:
                self.logger.error(f"安装 ModelScope 失败：{str(e)}")
                self.logger.info("请手动运行：pip install modelscope")
                return False
    
    def check_model_exists(self, model_path: str) -> bool:
        """检查模型文件是否存在"""
        if not model_path:
            return False
        
        path = Path(model_path)
        if path.exists():
            config_files = list(path.glob("*.json")) + list(path.glob("*.txt"))
            if config_files:
                self.logger.success(f"模型已存在：{model_path} ✓")
                return True
            else:
                self.logger.warning(f"模型目录存在但可能不完整：{model_path}")
                return False
        else:
            self.logger.info(f"模型不存在：{model_path}")
            return False
    
    def download_model(self, model_id: str, local_dir: str) -> bool:
        """使用 ModelScope 下载模型"""
        self.logger.info(f"开始下载模型：{model_id}")
        self.logger.info(f"保存路径：{local_dir}")
        
        try:
            result = subprocess.run(
                [
                    "modelscope", "download",
                    "--model", model_id,
                    "--local_dir", local_dir
                ],
                capture_output=False,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                self.logger.success(f"模型下载成功：{model_id} ✓")
                return True
            else:
                self.logger.error(f"模型下载失败：{model_id}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("模型下载超时")
            return False
        except Exception as e:
            self.logger.error(f"下载模型时出错：{str(e)}")
            return False
    
    def check_and_download_models(self, validator: ConfigValidator) -> Dict[str, bool]:
        """检查并下载所有需要的模型"""
        self.logger.header("模型检查与下载")
        self.validator = validator
        results = {}
        
        if not self.check_modelscope_installed():
            self.logger.error("请先安装 ModelScope")
            return {'modelscope_installed': False}
        
        results['modelscope_installed'] = True
        
        for model_name, model_info in self.models.items():
            self.logger.info(f"\n检查 {model_name} 模型...")
            
            model_path = None
            for config_path in model_info['config_paths']:
                path = self.get_model_path_from_config(config_path)
                if path:
                    model_path = path
                    break
            
            if not model_path:
                self.logger.warning(f"未找到 {model_name} 模型的配置路径")
                results[f'{model_name}_exists'] = False
                continue
            
            if self.check_model_exists(model_path):
                results[f'{model_name}_exists'] = True
            else:
                self.logger.info(f"需要下载 {model_name} 模型")
                response = input(f"是否下载 {model_name} 模型？(y/n): ").strip().lower()
                if response == 'y':
                    os.makedirs(model_path, exist_ok=True)
                    if self.download_model(model_info['model_id'], model_path):
                        results[f'{model_name}_exists'] = self.check_model_exists(model_path)
                    else:
                        results[f'{model_name}_exists'] = False
                else:
                    self.logger.warning(f"跳过 {model_name} 模型下载")
                    results[f'{model_name}_exists'] = False
        
        return results


class VectorStoreManager:
    """向量库管理器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.project_root = Path(__file__).parent.absolute()
        self.vector_store_dir = self.project_root / "vector_store"
        self.processed_data_dir = self.project_root / "processed_data"
    
    def check_vector_store_exists(self) -> bool:
        """检查向量库是否存在"""
        if self.vector_store_dir.exists() and any(self.vector_store_dir.iterdir()):
            self.logger.success("向量库已存在 ✓")
            return True
        else:
            self.logger.warning("向量库不存在或为空")
            return False
    
    def check_processed_data_exists(self) -> bool:
        """检查处理后的数据是否存在"""
        if self.processed_data_dir.exists() and any(self.processed_data_dir.iterdir()):
            self.logger.success("已处理的数据存在 ✓")
            return True
        else:
            self.logger.warning("未找到已处理的数据")
            return False
    
    def build_vector_store(self) -> bool:
        """构建向量库"""
        self.logger.header("构建向量库")
        
        if not self.check_processed_data_exists():
            self.logger.error("请先处理教材数据")
            self.logger.info("运行：python data_loader.py process \"./资料库\"")
            return False
        
        try:
            self.logger.info("开始构建向量库...")
            result = subprocess.run(
                [sys.executable, "vector_store.py", "--process-all", str(self.processed_data_dir)],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.success("向量库构建成功 ✓")
                return True
            else:
                self.logger.error("向量库构建失败")
                return False
        except Exception as e:
            self.logger.error(f"构建向量库时出错：{str(e)}")
            return False
    
    def run_check(self) -> bool:
        """运行向量库检查"""
        self.logger.header("向量库检查")
        
        if self.check_vector_store_exists():
            return True
        
        if self.check_processed_data_exists():
            self.logger.info("是否需要现在构建向量库？")
            response = input("是否构建向量库？(y/n): ").strip().lower()
            if response == 'y':
                return self.build_vector_store()
            else:
                self.logger.warning("跳过向量库构建")
                return False
        else:
            self.logger.error("请先处理教材数据")
            return False


class AppLauncher:
    """应用启动器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.project_root = Path(__file__).parent.absolute()
        self.chainlit_app = self.project_root / "chainlit_app.py"
    
    def check_chainlit_app(self) -> bool:
        """检查 Chainlit 应用文件"""
        if self.chainlit_app.exists():
            self.logger.success("Chainlit 应用文件存在 ✓")
            return True
        else:
            self.logger.error("未找到 chainlit_app.py")
            return False
    
    def launch(self) -> bool:
        """启动应用"""
        self.logger.header("启动应用")
        
        if not self.check_chainlit_app():
            return False
        
        try:
            self.logger.info("正在启动 Chainlit 应用...")
            self.logger.info("访问地址：http://localhost:8000")
            
            subprocess.run(
                [sys.executable, "-m", "chainlit", "run", str(self.chainlit_app), "-w"],
                cwd=self.project_root,
                capture_output=False
            )
            
            return True
        except KeyboardInterrupt:
            self.logger.info("应用已停止")
            return True
        except Exception as e:
            self.logger.error(f"启动应用失败：{str(e)}")
            return False


class DeploymentOrchestrator:
    """部署编排器"""
    
    def __init__(self):
        self.logger = Logger(log_file="logs/deploy.log")
        self.checker = DeploymentChecker(self.logger)
        self.validator = ConfigValidator(self.logger)
        self.model_downloader = ModelDownloader(self.logger)
        self.vector_manager = VectorStoreManager(self.logger)
        self.launcher = AppLauncher(self.logger)
    
    def run_deployment(self) -> bool:
        """运行完整部署流程"""
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}")
        print("=" * 70)
        print("AI 课程智能问答系统 - 一键部署工具".center(60))
        print("=" * 70)
        print(f"{Colors.ENDC}\n")
        
        self.logger.info(f"部署时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"项目路径：{self.checker.project_root}")
        
        env_checks = self.checker.run_all_checks()
        
        if not all(env_checks.values()):
            self.logger.error("环境检查未通过，请先修复环境问题")
            self._print_check_results(env_checks)
            return False
        
        config_results = self.validator.run_validation()
        
        if not all(config_results.values()):
            self.logger.error("配置验证未通过，请检查配置文件")
            self._print_check_results(config_results)
            return False
        
        self.logger.header("LLM 配置")
        llm_config = self.validator.config.get('qa_system', {}).get('llm', {})
        provider = llm_config.get('provider', 'api')
        api_key = llm_config.get('api_key', '')
        
        print(f"\n{Colors.OKCYAN}请选择 LLM 提供者：{Colors.ENDC}")
        print("  1) API (在线 API，如 DeepSeek)")
        print("  2) Local (本地 Ollama 模型)")
        print(f"  当前配置：{provider}")
        
        choice = input("\n请选择 (1/2，直接回车使用当前配置): ").strip()
        
        if choice == '1':
            provider = 'api'
            self.validator.update_provider('api')
            self.logger.info("使用 DeepSeek API")
            
            # 提示输入 API 模型名称
            current_model = llm_config.get('model_name', '')
            self.logger.info(f"当前模型名称：{current_model}")
            
            # 检查模型名称是否像本地 Ollama 模型（包含冒号）
            if current_model and ':' in current_model:
                self.logger.warning(f"模型名称 '{current_model}' 看起来是本地 Ollama 模型格式")
                self.logger.warning("API 模式应使用如 deepseek-chat 等模型名称")
                change_model = input("是否需要修改？(y/n，默认 y): ").strip().lower()
                if change_model != 'n':
                    api_model_input = input("请输入 API 模型名称：").strip()
                    if api_model_input:
                        self.validator.update_api_model(api_model_input)
                else:
                    api_model_input = None  # 保持原配置
            else:
                api_model_input = input("请输入 API 模型名称 (直接回车使用当前配置): ").strip()
                if api_model_input:
                    self.validator.update_api_model(api_model_input)
                elif not current_model:
                    self.logger.warning("未输入 API 模型名称，需要手动配置")
            
            api_key_input = input("请输入 DeepSeek API 秘钥：").strip()
            if api_key_input:
                if not self.validator.update_api_key(api_key_input):
                    self.logger.warning("API Key 更新失败，但仍可继续")
            else:
                self.logger.warning("未输入 API Key，使用配置文件中的值")
        elif choice == '2':
            provider = 'local'
            self.validator.update_provider('local')
            self.logger.info("使用本地 Ollama 模型")
            model_name = input("请输入本地 Ollama 模型名称：").strip()
            if model_name:
                self.validator.update_local_model(model_name)
                self.logger.success(f"本地模型已配置：{model_name} ✓")
            else:
                self.logger.warning("未输入模型名称，使用配置文件中的值")
            
            embedding_path = self.validator.config.get('vector_processing', {}).get('embedding', {}).get('local_path', '')
            if not embedding_path:
                self.logger.error("嵌入模型路径未配置（为空）")
                self.logger.error("请选择在线模型或配置本地嵌入模型路径")
                return False
            elif not Path(embedding_path).exists():
                self.logger.error(f"嵌入模型路径不存在：{embedding_path}")
                self.logger.error("请检查路径配置或创建相应目录")
                return False
            else:
                self.logger.success(f"嵌入模型路径验证通过：{embedding_path} ✓")
        else:
            if provider == 'api':
                if api_key and api_key.startswith('sk-'):
                    self.logger.success(f"API Key 已配置：{api_key[:8]}...{api_key[-4:]} ✓")
                else:
                    self.logger.warning("API Key 未正确配置")
                
                # 检查模型名称是否适合 API 使用
                model_name = llm_config.get('model_name', '')
                self.logger.info(f"当前 API 模型名称：{model_name}")
                
                # 只检查是否像本地 Ollama 模型（包含冒号）
                if model_name and ':' in model_name:
                    self.logger.warning(f"模型名称 '{model_name}' 看起来是本地 Ollama 模型格式")
                    self.logger.warning("API 模式应使用如 deepseek-chat 等模型名称")
                    change_model = input("是否要修改模型名称？(y/n，默认 y): ").strip().lower()
                    if change_model != 'n':
                        new_model = input("请输入新的 API 模型名称：").strip()
                        if new_model:
                            self.validator.update_api_model(new_model)
                        else:
                            self.logger.warning("未输入模型名称，保持原配置")
            else:
                model_name = llm_config.get('model_name', '')
                self.logger.info(f"使用本地模型：{model_name}")
                
                embedding_path = self.validator.config.get('vector_processing', {}).get('embedding', {}).get('local_path', '')
                if not embedding_path:
                    self.logger.error("嵌入模型路径未配置（为空）")
                    self.logger.error("请选择在线模型或配置本地嵌入模型路径")
                    return False
                elif not Path(embedding_path).exists():
                    self.logger.error(f"嵌入模型路径不存在：{embedding_path}")
                    self.logger.error("请检查路径配置或创建相应目录")
                    return False
                else:
                    self.logger.success(f"嵌入模型路径验证通过：{embedding_path} ✓")
        
        model_results = self.model_downloader.check_and_download_models(self.validator)
        if not all(model_results.values()):
            self.logger.warning("部分模型未就绪，可能影响功能")
            self._print_check_results(model_results)
            
            self.logger.info(f"模型默认下载路径：")
            for model_name, model_info in self.model_downloader.models.items():
                model_id = model_info.get('model_id', model_name)
                model_path = f"D:/models/{model_id}"
                self.logger.info(f"  {model_name} 模型：{model_path}")
            
            response = input("是否下载并安装模型？(y/n): ").strip().lower()
            if response == 'y':
                self.logger.info("开始下载模型...")
                model_results = self.model_downloader.check_and_download_models(self.validator)
                if not all(model_results.values()):
                    self.logger.error("模型下载失败，部分模型仍未就绪")
                    self._print_check_results(model_results)
                    return False
                self.logger.success("模型下载完成 ✓")
            else:
                self.logger.warning("跳过模型下载，部分功能可能不可用")
                return False
        
        if not self.vector_manager.run_check():
            self.logger.warning("向量库未就绪，问答功能可能受限")
            response = input("是否继续启动？(y/n): ").strip().lower()
            if response != 'y':
                return False
        
        return self.launcher.launch()
    
    def _print_check_results(self, results: Dict[str, bool]):
        """打印检查结果"""
        print(f"\n{Colors.WARNING}检查结果：{Colors.ENDC}")
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            color = Colors.OKGREEN if passed else Colors.FAIL
            print(f"  {color}{status} {check}{Colors.ENDC}")


def main():
    """主函数"""
    try:
        orchestrator = DeploymentOrchestrator()
        success = orchestrator.run_deployment()
        
        if success:
            sys.exit(0)
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}部署失败，请检查上述错误信息{Colors.ENDC}\n")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}部署被用户中断{Colors.ENDC}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}发生未知错误：{str(e)}{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
