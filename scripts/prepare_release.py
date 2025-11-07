#!/usr/bin/env python
"""
准备发布到GitHub的干净项目结构
Prepare clean project structure for GitHub release
"""

import os
import shutil
from pathlib import Path

# 源目录和目标目录
SOURCE_DIR = Path(".")
TARGET_DIR = Path("YanNian-Mol")

# 必须包含的文件和目录
ESSENTIAL_FILES = [
    # 核心包文件
    "lifespan_predictor/",
    
    # 配置文件
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    
    # 文档
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "VERSION",
    
    # 依赖
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-frozen.txt",
    
    # 示例和文档
    "notebooks/",
    "examples/",
    "docs/",
    
    # 测试
    "tests/",
    
    # 脚本
    "scripts/",
    
    # 配置
    ".gitignore",
    ".pre-commit-config.yaml",
    "pytest.ini",
    "mypy.ini",
    "Makefile",
    
    # CI/CD
    ".github/",
    
    # 数据示例（如果有小的示例数据）
    "data/sample/",
]

# 需要排除的文件和目录
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "env/",
    "venv/",
    "ENV/",
    "build/",
    "dist/",
    "*.egg-info/",
    ".pytest_cache/",
    ".mypy_cache/",
    "htmlcov/",
    ".coverage",
    ".git/",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",
    # 大文件和处理后的数据
    "processed_graph_features/",
    "processed_fingerprints/",
    "checkpoints/",
    "logs/",
    "runs/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.npy",
    "*.pkl",
    "*.pickle",
    # 原始notebook文件（已经有重构后的版本）
    "run_featurizer_lifespan.ipynb",
    "run_fingerprint_lifespan_simplified.ipynb",
    "LifespanPredictClass.ipynb",
    "LifespanPredict.ipynb",
    "GetPrediction_From_Checkpoint_deepGraphh.ipynb",
    "05practice.ipynb",
    "fm4m_example.ipynb",
    # 其他项目文件夹
    "MDF-DTA-A-Multi-Dimensional-Fusion-Approach-for-Drug-Target-Binding-Affinity-Prediction-main/",
    "Geroprotectors-Project-INGER-main/",
    # 临时文件
    "validation_report.json",
    "requirements-pinned.txt",
    "TASK_*_SUMMARY.md",
    ".git_commit_msg.tmp",
]


def should_exclude(path: Path) -> bool:
    """检查路径是否应该被排除"""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            if pattern[:-1] in path.parts:
                return True
        elif pattern.startswith("*."):
            if path_str.endswith(pattern[1:]):
                return True
        elif pattern in path_str:
            return True
    return False


def copy_directory(src: Path, dst: Path):
    """递归复制目录，排除不需要的文件"""
    if not dst.exists():
        dst.mkdir(parents=True)
    
    for item in src.iterdir():
        src_path = src / item.name
        dst_path = dst / item.name
        
        if should_exclude(src_path):
            continue
        
        if src_path.is_dir():
            copy_directory(src_path, dst_path)
        else:
            print(f"复制: {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)


def main():
    """主函数"""
    print("=" * 80)
    print("准备YanNian-Mol发布包")
    print("Preparing YanNian-Mol Release Package")
    print("=" * 80)
    
    # 创建目标目录
    if TARGET_DIR.exists():
        print(f"\n清理现有目录: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    
    TARGET_DIR.mkdir(parents=True)
    print(f"✓ 创建目录: {TARGET_DIR}")
    
    # 复制必要文件
    print("\n复制必要文件...")
    
    for item in ESSENTIAL_FILES:
        src_path = SOURCE_DIR / item
        
        if not src_path.exists():
            print(f"⚠ 跳过不存在的: {item}")
            continue
        
        if src_path.is_dir():
            dst_path = TARGET_DIR / item
            print(f"\n复制目录: {item}")
            copy_directory(src_path, dst_path)
        else:
            dst_path = TARGET_DIR / item
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"复制文件: {item}")
            shutil.copy2(src_path, dst_path)
    
    # 创建示例数据目录结构
    sample_data_dir = TARGET_DIR / "data" / "sample"
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据说明文件
    readme_content = """# Sample Data

This directory should contain small sample datasets for testing and demonstration.

## Expected Files

- `sample_train.csv` - Small training dataset (10-20 compounds)
- `sample_test.csv` - Small test dataset (5-10 compounds)

## Format

CSV files should have columns:
- `SMILES` - Molecular SMILES string
- `Life_extended` - Binary label (0 or 1)

## Usage

```python
from lifespan_predictor.data.preprocessing import load_and_clean_csv

# Load sample data
df = load_and_clean_csv('data/sample/sample_train.csv')
```

## Note

Full datasets are not included in the repository due to size.
Please refer to the main README for data sources and preparation instructions.
"""
    
    with open(sample_data_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✓ 创建示例数据目录: {sample_data_dir}")
    
    # 统计文件
    total_files = sum(1 for _ in TARGET_DIR.rglob("*") if _.is_file())
    total_dirs = sum(1 for _ in TARGET_DIR.rglob("*") if _.is_dir())
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n✓ 目标目录: {TARGET_DIR.absolute()}")
    print(f"✓ 文件数量: {total_files}")
    print(f"✓ 目录数量: {total_dirs}")
    
    print("\n下一步:")
    print("1. 检查 YanNian-Mol 目录内容")
    print("2. 添加示例数据到 data/sample/ (可选)")
    print("3. 进入目录: cd YanNian-Mol")
    print("4. 初始化git: git init")
    print("5. 添加远程: git remote add origin https://github.com/yuzhounaut/YanNian-Mol.git")
    print("6. 提交并推送: git add . && git commit -m 'Initial commit' && git push -u origin main")


if __name__ == "__main__":
    main()
