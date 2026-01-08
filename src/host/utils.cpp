#include "host_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



bool load_sudoku(const char* filename, int* board) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Cannot open file");
        return false;
    }

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int value;
            if (fscanf(file, "%d", &value) != 1) {
                fprintf(stderr, "Bad format at row %d col %d\n", row, col);
                fclose(file);
                return false;
            }

            if (value < 0 || value > N) {
                fprintf(stderr, "Invalid value %d at (%d,%d)\n",
                    value, row, col);
                fclose(file);
                return false;
            }

            board[row * N + col] = value;
        }
    }

    fclose(file);
    return true;
}

void print_sudoku(const int* board) {

    for (int row = 0; row < N; row++) {
        if (row % SQRT_N == 0)
            printf("+-------+-------+-------+\n");

        for (int col = 0; col < N; col++) {
            if (col % SQRT_N == 0) printf("| ");
            int v = board[row * N + col];
            if(v!=0)
                printf("%d ", v);
            else
                printf(". ");
        }
        printf("|\n");
    }
    printf("+-------+-------+-------+\n");
}


bool is_safe(int *board, int index, int num) {
    int row = index / N;
    int col = index % N;

    
    for (int i = 0; i < N; i++) {
        if (board[row * N + i] == num) return false;
        if (board[i * N + col] == num) return false;
    }

    
    int startRow = (row / SQRT_N) * SQRT_N;
    int startCol = (col / SQRT_N) * SQRT_N;

    for (int r = 0; r < SQRT_N; r++) {
        for (int c = 0; c < SQRT_N; c++) {
            if (board[(startRow + r) * 9 + (startCol + c)] == num)
                return false;
        }
    }

    return true;
}

bool solve_sudoku(int *board) {
    int index = -1;

    
    for (int i = 0; i < SIZE; i++) {
        if (board[i] == 0) {
            index = i;
            break;
        }
    }


    if (index == -1)
        return true;

 
    for (int num = 1; num <= N; num++) {
        if (is_safe(board, index, num)) {
            board[index] = num;

            if (solve_sudoku(board))
                return true;

            // Backtracking
            board[index] = 0;
        }
    }

    return false;
}
