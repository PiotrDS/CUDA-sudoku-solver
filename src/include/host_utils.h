#ifndef HOST_UTILS_H
#define HOST_UTILS_H
#include <math.h>
// constants
#define N 9
#define SIZE (N*N)
#define SQRT_N ((int)sqrt(N))

bool load_sudoku(const char* filename,
	int* board);

void print_sudoku(const int* board);

bool is_safe(int* board, 
	int index, 
	int num);

bool solve_sudoku(int* board);

#endif