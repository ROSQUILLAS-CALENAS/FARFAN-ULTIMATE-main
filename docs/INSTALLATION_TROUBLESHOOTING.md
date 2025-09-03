# EGW Query Expansion Installation Troubleshooting Guide

This comprehensive guide provides step-by-step troubleshooting instructions for common installation issues with the EGW (Entropic Gromov-Wasserstein) Query Expansion system.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Python Version Issues](#python-version-issues)
3. [Virtual Environment Problems](#virtual-environment-problems)
4. [Dependency Resolution Issues](#dependency-resolution-issues)
5. [Platform-Specific Issues](#platform-specific-issues)
6. [Core Library Problems](#core-library-problems)
7. [GPU and CUDA Issues](#gpu-and-cuda-issues)
8. [Memory and Performance Issues](#memory-and-performance-issues)
9. [Configuration Issues](#configuration-issues)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Diagnosis

Run the built-in diagnostic tool for an automated environment analysis:

```bash
python -m egw_query_expansion.cli.troubleshoot
```

Or use the standalone validation script:

```bash
python validate_installation.py
```

## Python Version Issues

### Problem: Unsupported Python Version

**Symptoms:**
- Import errors during installation
- Package compatibility warnings
- Build failures with cryptic error messages

**Supported Versions:**
- Python 3.8 - 3.12 (recommended: 3.10 or 3.11)
- Python 3.13+ not fully supported due to some dependency limitations

**Solution:**

1. **Check your Python version:**
   ```bash
   python --version
   python3 --version
   ```

2. **Install supported Python version using pyenv (recommended):**
   ```bash
   # Install pyenv if not already installed
   curl https://pyenv.run | bash
   
   # Install Python 3.11
   pyenv install 3.11.7
   pyenv global 3.11.7
   
   # Verify installation
   python --version
   ```

3. **Alternative: Use conda:**
   ```bash
   conda create -n egw-env python=3.11
   conda activate egw-env
   ```

### Problem: Multiple Python Versions Conflict

**Symptoms:**
- Wrong Python executable being used
- Packages installed in wrong Python environment
- Import errors despite successful installation

**Solution:**

1. **Identify Python installations:**
   ```bash
   which python
   which python3
   ls -la $(which python)
   ls -la $(which python3)
   ```

2. **Use explicit Python path:**
   ```bash
   /usr/bin/python3.11 -m pip install -r requirements.txt
   ```

3. **Create alias for consistency:**
   ```bash
   echo "alias python=/usr/bin/python3.11" >> ~/.bashrc
   source ~/.bashrc
   ```

## Virtual Environment Problems

### Problem: Virtual Environment Not Working

**Symptoms:**
- Packages install globally instead of locally
- Import errors despite installation
- Permission errors during installation

**Solution:**

1. **Create fresh virtual environment:**
   ```bash
   # Remove existing environment
   rm -rf venv/
   
   # Create new environment
   python -m venv venv
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Verify activation
   which python
   which pip
   ```

2. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Problem: Virtual Environment Path Issues

**Symptoms:**
- Virtual environment not found
- Wrong Python executable used
- Environment variables not set

**Solution:**

1. **Use absolute paths:**
   ```bash
   python -m venv /absolute/path/to/venv
   source /absolute/path/to/venv/bin/activate
   ```

2. **Set environment variables:**
   ```bash
   export VIRTUAL_ENV=/path/to/venv
   export PATH=$VIRTUAL_ENV/bin:$PATH
   ```

3. **Verify environment:**
   ```bash
   echo $VIRTUAL_ENV
   echo $PATH
   python -c "import sys; print(sys.prefix)"
   ```

## Dependency Resolution Issues

### Problem: Conflicting Package Versions

**Symptoms:**
- pip reports version conflicts
- Import errors for specific packages
- Different packages requiring incompatible versions

**Solution:**

1. **Use dependency resolver:**
   ```bash
   pip install --upgrade pip  # Ensure pip >= 20.3
   pip install -r requirements.txt --use-feature=2020-resolver
   ```

2. **Install core dependencies first:**
   ```bash
   pip install numpy>=1.24.0 scipy>=1.11.0
   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
   pip install faiss-cpu>=1.7.4
   pip install transformers>=4.35.0
   pip install POT>=0.9.1
   pip install -r requirements.txt
   ```

3. **Use constraints file (create requirements-constraints.txt):**
   ```bash
   # Create constraints file
   pip freeze > requirements-constraints.txt
   
   # Install with constraints
   pip install -r requirements.txt -c requirements-constraints.txt
   ```

### Problem: Package Not Found

**Symptoms:**
- "No matching distribution found"
- Package name not recognized
- Platform compatibility issues

**Solution:**

1. **Check package availability:**
   ```bash
   pip search <package-name>  # If search is available
   pip index versions <package-name>
   ```

2. **Use alternative package sources:**
   ```bash
   # Use conda-forge for difficult packages
   conda install -c conda-forge <package-name>
   
   # Use alternative PyPI index
   pip install -i https://pypi.org/simple/ <package-name>
   ```

3. **Install from source:**
   ```bash
   pip install git+https://github.com/owner/repo.git
   ```

### Problem: Build Dependencies Missing

**Symptoms:**
- Compilation errors during pip install
- "Microsoft Visual C++ required" (Windows)
- "gcc not found" (Linux)

**Solution:**

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Alpine
apk add build-base python3-dev
```

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Or install via Homebrew
brew install gcc
```

**Windows:**
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use conda environment
conda install m2w64-toolchain
```

## Platform-Specific Issues

### Windows Issues

#### Problem: Long Path Names

**Solution:**
```bash
# Enable long paths in Windows
git config --global core.longpaths true

# Or use short virtual environment path
python -m venv C:\venv
```

#### Problem: Permission Errors

**Solution:**
```bash
# Run as administrator or use user installation
pip install --user -r requirements.txt

# Or modify permissions
icacls "C:\Python311" /grant Users:F /T
```

### macOS Issues

#### Problem: Apple Silicon Compatibility

**Solution:**
```bash
# Install specific versions for ARM64
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Use conda for better ARM64 support
conda install pytorch torchvision -c pytorch

# Force x86_64 if needed
arch -x86_64 pip install <package>
```

#### Problem: SSL Certificate Errors

**Solution:**
```bash
# Update certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or install certificates manually
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Linux Issues

#### Problem: System Package Conflicts

**Solution:**
```bash
# Use isolated environment
python -m venv venv --system-site-packages=false

# Or use containerized approach
docker run -it python:3.11 bash
```

#### Problem: Missing System Libraries

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev libopenblas-dev liblapack-dev

# For FAISS
sudo apt-get install libfaiss-dev

# For image processing
sudo apt-get install libopencv-dev python3-opencv
```

## Core Library Problems

### FAISS Installation Issues

#### Problem: FAISS Won't Install

**Solution:**
```bash
# Try CPU version first
pip install faiss-cpu

# For GPU support
pip install faiss-gpu

# From conda-forge (often more reliable)
conda install -c conda-forge faiss-cpu

# Build from source if needed
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_PYTHON=ON
make -C build -j faiss swigfaiss
cd build/faiss/python
python setup.py install
```

### PyTorch Installation Issues

#### Problem: PyTorch Version Conflicts

**Solution:**
```bash
# Install specific PyTorch version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Check compatibility
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### POT (Python Optimal Transport) Issues

#### Problem: POT Build Failures

**Solution:**
```bash
# Install dependencies first
pip install numpy Cython

# Try specific version
pip install POT==0.9.1

# From conda
conda install -c conda-forge pot

# Build from source
pip install git+https://github.com/PythonOT/POT.git
```

## GPU and CUDA Issues

### Problem: CUDA Not Detected

**Solution:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install CUDA toolkit
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: GPU Memory Issues

**Solution:**
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Use CPU-only versions if GPU memory insufficient
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu

# Adjust batch sizes in configuration
```

## Memory and Performance Issues

### Problem: Installation Runs Out of Memory

**Solution:**
```bash
# Install packages one by one
pip install numpy
pip install scipy
pip install torch
# ... continue with individual packages

# Use pip with no cache
pip install --no-cache-dir -r requirements.txt

# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Problem: Slow Installation

**Solution:**
```bash
# Use multiple workers
pip install -r requirements.txt --use-feature=parallel-install

# Use faster index
pip install -r requirements.txt -i https://pypi.douban.com/simple/

# Install pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

## Configuration Issues

### Problem: Configuration Files Not Found

**Solution:**
```bash
# Check configuration path
python -c "import egw_query_expansion; print(egw_query_expansion.__file__)"

# Create default configuration
mkdir -p egw_query_expansion/configs/
cp egw_query_expansion/configs/default_config.yaml.template egw_query_expansion/configs/default_config.yaml

# Set environment variable
export EGW_CONFIG_PATH=/path/to/config
```

### Problem: Model Download Failures

**Solution:**
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/cache

# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-medium')"

# Use offline mode if needed
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Advanced Troubleshooting

### Enable Debug Logging

```bash
export EGW_DEBUG=1
export PYTHONPATH=/path/to/egw_query_expansion:$PYTHONPATH
python validate_installation.py
```

### Dependency Analysis

```bash
# Check installed packages
pip list
pip show <package-name>

# Check for conflicts
pip check

# Generate dependency tree
pip install pipdeptree
pipdeptree

# Check import paths
python -c "import sys; print('\n'.join(sys.path))"
```

### Clean Installation

```bash
# Remove all packages and start fresh
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt

# Or recreate virtual environment
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Container-based Installation

If all else fails, use Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

CMD ["python", "validate_installation.py"]
```

Build and run:
```bash
docker build -t egw-query-expansion .
docker run -it egw-query-expansion
```

## Getting Help

If you continue to experience issues:

1. **Run the diagnostic tool:**
   ```bash
   python -m egw_query_expansion.cli.troubleshoot --verbose --export-report
   ```

2. **Check the GitHub issues:** [Repository Issues](https://github.com/your-repo/egw-query-expansion/issues)

3. **Provide diagnostic information:**
   - Python version (`python --version`)
   - Operating system and version
   - Virtual environment details
   - Complete error messages
   - Output from diagnostic tool

4. **Create a minimal reproducible example** demonstrating the issue

Remember: Most installation issues stem from Python version incompatibilities, virtual environment problems, or missing system dependencies. The diagnostic CLI tool can help identify the specific issue in your environment.