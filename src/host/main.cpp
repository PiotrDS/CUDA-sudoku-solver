#include "device_utils.h"
#include "host_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {

    const char* path_to_file;
    int depth = 18;
    int num_blocks = 512;
    int num_threads = 256;

    int board[SIZE];


    for (int i = 1; i < argc; i++) {

        if (strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            path_to_file = argv[++i];
            if (!load_sudoku(path_to_file, board)) {
                printf("Error: Could't load sudoku from path %s\n", path_to_file);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            depth = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            num_blocks = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--thr") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf(
                "Usage: sudoku_solver.exe\n"
                "  --p \t Path to file with sudoku table to solve (char, obligatory!)\n"
                "  --d \t Depth of BFS search (int, default: 18)\n"
                "  --nb \t Number of blocks (int, default: 512)\n"
                "  --nt \t Number of dimensions (int, default: 256)\n"

            );
            return 0;
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            printf("type sudoku_solver.exe --help for more information\n");
            return 1;
        }
    }


    int* d_boards_src;
    int* d_boards_dst;
    int* d_empty;
    int* d_emptyCount;
    int* d_counter;
    int* d_finished;
    int* d_solution;

	if (!load_sudoku(path_to_file, board)) {
		printf("Error: Could't load sudoku from path %s\n", path_to_file);
		return 1;
	}


    cudaMalloc(&d_boards_src, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_boards_dst, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_empty, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_emptyCount, MAX_BFS_BOARDS * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_finished, sizeof(int));
    cudaMalloc(&d_solution, SIZE * sizeof(int));

    cudaMemcpy(d_boards_src, board, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int currentBoards = 1;

    float bfs_time = 0.0f;

    for (int lvl = 0; lvl < depth; lvl++) {

        cudaMemset(d_counter, 0, sizeof(int));

        bfs_time = bfs_time + cuda_sudoku_BFS (
            num_blocks,
            num_threads,
            d_boards_src,
            d_boards_dst,
            currentBoards,
            d_counter,
            d_empty,
            d_emptyCount
            );

        cudaDeviceSynchronize();

        cudaMemcpy(&currentBoards, d_counter,
            sizeof(int), cudaMemcpyDeviceToHost);

        if (currentBoards > MAX_BFS_BOARDS)
            currentBoards = MAX_BFS_BOARDS;

        std::swap(d_boards_src, d_boards_dst);
    }
    printf("\n===== GPU BFS TIMING =====\n");
    printf("Total:                   %.3f ms\n", bfs_time);
    printf("=========================\n\n");

    cudaMemset(d_finished, 0, sizeof(int));

    int num_block_new = (currentBoards + num_threads -1) / num_threads;
    cuda_sudoku_backtrack (
        num_block_new,
        num_threads,
        d_boards_src,
        currentBoards,
        d_empty,
        d_emptyCount,
        d_finished,
        d_solution
        );

    cudaDeviceSynchronize();

    cudaMemcpy(board, d_solution, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    print_sudoku(board);

    cudaFree(d_boards_src);
    cudaFree(d_boards_dst);
    cudaFree(d_empty);
    cudaFree(d_emptyCount);
    cudaFree(d_counter);
    cudaFree(d_finished);
    cudaFree(d_solution);

    return 0;
}
