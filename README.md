# ParallelPricer

### Monte Carlo simulation for options pricing with CUDA C++ parallelism optimization

### System Requirements
- Windows 10 version 2004 or higher, or Windows 11
- CUDA enabled NVIDIA GPU

### Installation
1. Install Windows Subsystem for Linux (WSL)
   1. On Windows Powershell
      ```bash
      wsl --install
      ```
   2. Restart PC
   3. Create Ubuntu account
2. Update Ubuntu
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Install build stuff
   ```bash
   sudo apt install build-essential
   ```
4. Install CUDA on WSL
   1. Add NVIDIA pkg repo
      ```bash
      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
      sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
      sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
      sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
      sudo apt-get update
      ```
   2. Install CUDA
      ```bash
      sudo apt-get install cuda
      ```
   3. Environment variables (bashrc)
      ```bash
      export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      ```
   4. Apply changes
      ```bash
      source ~/.bashrc
      ```
5. Verify CUDA
   ```bash
   nvcc --version
   ```
### Setup
1. Clone this repo (assuming you haven't already)
2. Build the project
   ```bash
   make
   ```
3. Run simulation
   ```bash
   ./monte_carlo_sim
   ```