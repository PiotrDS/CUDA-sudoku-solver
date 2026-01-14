#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H


#define N 9
#define SUB 3
#define BOARD_SIZE 81
#define MAX_BFS_BOARDS 40000
#define NUM_BLOCKS 512
#define NUM_THREADS 256



void cuda_sudoku_backtrack(int* boards,
    int num_boards,
    int* empty_spaces,
    int* empty_count,
    int* finished,
    int* solution);

void cuda_sudoku_BFS(int* old_boards,
    int* new_boards,
    int old_count,
    int* new_count,
    int* empty_spaces,
    int* empty_count);

#endif