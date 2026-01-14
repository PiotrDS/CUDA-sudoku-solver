#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H


#define N 9
#define SIZE (N*N)
#define SQRT_N ((int)sqrt(N))
#define MAX_BFS_BOARDS 50000




void cuda_sudoku_backtrack(
    int num_blocks,
    int num_threads,
    int* boards,
    int numBoards,
    int* emptySpaces,
    int* emptyCount,
    int* finished,
    int* solved
);

float cuda_sudoku_BFS(
    int num_blocks,
    int num_threads,
    int* oldBoards,
    int* newBoards,
    int oldCount,
    int* newCount,
    int* emptySpaces,
    int* emptyCount
);


#endif