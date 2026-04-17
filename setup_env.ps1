# setup_env.ps1
# Automated environment setup for USD Digital Twin Playground

Write-Host "--- Starting Environment Setup ---" -ForegroundColor Cyan

# 1. Update pip
python -m pip install --upgrade pip

# 2. Install PyTorch with CUDA 12.1 (recommended for gsplat/3DGRUT)
Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Group 1: OpenUSD Standalone
Write-Host "Installing usd-core..." -ForegroundColor Yellow
pip install usd-core

# 4. Install Group 3: gsplat (This may take a while as it compiles CUDA extensions)
Write-Host "Installing gsplat..." -ForegroundColor Yellow
pip install gsplat

# 5. Install Group 4: Nerfstudio
Write-Host "Installing nerfstudio..." -ForegroundColor Yellow
pip install nerfstudio

# 6. Install remaining requirements
Write-Host "Installing remaining dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "--- Setup Complete ---" -ForegroundColor Green
