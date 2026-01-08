#include "device_utils.h"
#include "host_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
	int board[SIZE];


	if (!load_sudoku("sudoku / sudoku_1.txt", board)) {
		printf("blad wczytywania\n");
		return 1;
	}

	if (solve_sudoku(board)) {
		printf("Rozwiazane Sudoku:\n");
		print_sudoku(board);
	}
	else {
		printf("Brak rozwiazania\n");
	}
	return 0;
}