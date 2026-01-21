# CUDA-sudoku-solver: GPU-Accelerated Sudoku Solver 🚀📊

This program implements the **Sudoku Solver** with support for both **GPU (CUDA)** and **CPU** execution. It allows you to solve provided sudoku table.

## Getting Started

### Prerequisites
Download and install the CUDA Toolkit for your corresponding platform. For system requirements and installation instructions of [cuda toolkit](https://developer.nvidia.com/cuda-downloads), please refer to the [Linux Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Getting the CUDA-kmeans-clustering
Using git clone the repository of CUDA-kmeans-clustering using the command below.
```bash
git clone https://github.com/PiotrDS/CUDA-sudoku-solver.git
```
### Building CUDA-sudoku-solver
The CUDA-sudoku-solver are built using CMake. Follow the instructions below:

Ensure that CMake (version 3.20 or later) is installed. Install it using your package manager if necessary.

Navigate to the root of the cloned repository and create a build directory:
```bash
mkdir build && cd build
```

Configure the project with CMake:
```
cmake ..
cmake --build .
cd Debug
```

Run the program with the following command line arguments:

```bash
sudoku_solver.exe --p <path to sudoku table>
```

### Required arguments

* `--p` : Path to file with sudoku table to solve (example sudoku table files are avaiable under sudoku directory)

### Optional arguments

*  `--d`: Depth of BFS search (int, default: 18)
*   `-nb`: Number of blocks (int, default: 512)
*  `--nt`: Number of threads (int, default: 256)
*  `--gpu`: Wheter to use gpu (gpu=1) or cpu (gpu=0) (int, default: 1)

### Example

Run with 15 steps depth, number of blocks = 256 and number of threads =128 on GPU:

```bash
sudoku_solver.exe --p <path to sudoku table> --d 15 --nb 256 --nt 128 --gpu 1
```

Run the CPU version:

```bash
sudoku_solver.exe --p <path to sudoku table> --d 15 --nb 256 --nt 128 --gpu 0
```

---

## Output

* Solved sudoku table printed in console
---
