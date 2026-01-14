#include "device_utils.h"
#include "host_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
	int board[SIZE];


	if (!load_sudoku("C:/Users/Tomek/source/repos/CUDA-sudoku-solver/sudoku/sudoku_1.txt", board)) {
		printf("blad wczytywania\n");
		return 1;
	}

	//if (solve_sudoku(board)) {
	//	printf("Rozwiazane Sudoku:\n");
	//	print_sudoku(board);
	//}
	//else {
	//	printf("Brak rozwiazania\n");
	//}
	//return 0;


    int* d_boardsA, * d_boardsB;
    int* d_empty, * d_emptyCount;
    int* d_counter, * d_finished, * d_solution;

    cudaMalloc(&d_boardsA, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_boardsB, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_empty, MAX_BFS_BOARDS * SIZE * sizeof(int));
    cudaMalloc(&d_emptyCount, MAX_BFS_BOARDS * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_finished, sizeof(int));
    cudaMalloc(&d_solution, SIZE * sizeof(int));

    cudaMemcpy(d_boardsA, board, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int zero = 0, one = 1;
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    cuda_sudoku_BFS(
        d_boardsA, d_boardsB, 1, d_counter, d_empty, d_emptyCount
        );

    int h_count;
    cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_finished, &zero, sizeof(int), cudaMemcpyHostToDevice);

    cuda_sudoku_backtrack(
        d_boardsB, h_count, d_empty, d_emptyCount, d_finished, d_solution
        );

    int h_solution[81];
    cudaMemcpy(h_solution, d_solution, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Solved Sudoku:\n");
    print_sudoku(h_solution);

    return 0;
}