"""
极简环境快照脚本 - 导出当前环境所有依赖
"""
import sys
import subprocess
import platform
import json

def get_system_info():
    """获取系统及硬件信息"""
    info = {
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "Machine": platform.machine(),
    }
    
    # 尝试获取 GPU 信息
    try:
        import torch
        info["Torch"] = torch.__version__
        info["CUDA Available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["GPU Count"] = torch.cuda.device_count()
            info["GPU Name"] = torch.cuda.get_device_name(0)
            info["CUDA Version"] = torch.version.cuda
    except ImportError:
        info["Torch"] = "Not Installed"
        
    return info

def generate_full_requirements():
    """获取所有已安装包并生成 requirements.txt"""
    try:
        # 使用 pip list --format=json 获取最准确的数据
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=True
        )
        packages = json.loads(result.stdout)
        
        # 排序并格式化
        sorted_pkgs = sorted(packages, key=lambda x: x["name"].lower())
        
        content = "# Auto-generated requirements.txt\n"
        content += f"# Generated on: {platform.platform()} | Python {platform.python_version()}\n\n"
        
        count = 0
        for pkg in sorted_pkgs:
            # 过滤掉 pip/setuptools/wheel 等基础工具（可选，如需保留可注释掉下一行）
            if pkg["name"].lower() in ["pip", "setuptools", "wheel"]:
                continue
            
            content += f"{pkg['name']}=={pkg['version']}\n"
            count += 1
            
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(content)
            
        return count, sorted_pkgs[:5] # 返回总数和前5个包用于展示
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return 0, []

def main():
    print("=" * 50)
    print("📦 环境快照生成器 (All Packages)")
    print("=" * 50)

    # 1. 系统信息
    print("\n🖥️  系统环境:")
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"   {k}: {v}")

    # 2. 生成全量依赖
    print("\n💾 正在扫描所有已安装包...")
    count, sample = generate_full_requirements()
    
    if count > 0:
        print(f"   ✅ 成功生成 requirements.txt")
        print(f"   📊 共包含 {count} 个包")
        print(f"   🔍 示例前5个: {[p['name'] for p in sample]}")
    else:
        print("   ❌ 未能生成文件，请检查 pip 是否正常")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()