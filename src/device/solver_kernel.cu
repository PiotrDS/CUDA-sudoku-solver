#include "device_utils.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdint>


__device__
bool valid_board(const int* board, int changed_idx) {

    int val = board[changed_idx];
    if (val < 1 || val > 9) return false;

    int r = changed_idx / 9;
    int c = changed_idx % 9;

    uint16_t seen_mask;

    // check row
    seen_mask = 0;
    for (int i = 0; i < 9; i++) {
        int v = board[r * 9 + i];
        if (v) {
            uint16_t bit_mask = 1 << (v - 1);
            if (seen_mask & bit_mask)
                return false;
            seen_mask |= bit_mask;
        }
    }

    // check column 
    seen_mask = 0;
    for (int i = 0; i < 9; i++) {
        int v = board[i * 9 + c];
        if (v) {
            uint16_t bit_mask = 1 << (v - 1);
            if (seen_mask & bit_mask)
                return false;
            seen_mask |= bit_mask;
        }
    }

    // check box
    seen_mask = 0;
    int b_r = (r / 3) * 3;
    int b_c = (c / 3) * 3;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int v = board[(b_r + i) * 9 + (b_c + j)];
            if (v) {
                uint16_t b = 1 << (v - 1);
                if (seen_mask & b) return false;
                seen_mask |= b;
            }
        }
    }

    return true;
}


__global__
void sudoku_backtrack(
    int* boards,
    int numBoards,
    int* emptySpaces,
    int* emptyCount,
    int* finished,
    int* solution) {

    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (g_idx < numBoards && atomicAdd(finished, 0) == 0) {

        int* board = boards + g_idx * BOARD_SIZE;
        int* empties = emptySpaces + g_idx * BOARD_SIZE;
        int total = emptyCount[g_idx];

        int idx = 0;

        while (idx >= 0 && idx < total) {

            int pos = empties[idx];
            int& cell = board[pos];

            cell++;

            if (cell <= 9 && valid_board(board, pos)) {
                idx++;
            }
            else {
                if (cell >= 9) {
                    cell = 0;
                    idx--;
                }
            }
        }

        if (idx == total) {
            if (atomicCAS(finished, 0, 1) == 0) {
                for (int i = 0; i < BOARD_SIZE; i++)
                    solution[i] = board[i];
            }
            return;
        }

        g_idx += blockDim.x * gridDim.x;
    }
}


// =======================================================
// BFS KERNEL (GENERATE BOARDS)
// =======================================================

__global__
void sudoku_BFS(
    int* oldBoards,
    int* newBoards,
    int oldCount,
    int* newCount,
    int* emptySpaces,
    int* emptyCount) {

    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= oldCount) return;

    int* board = oldBoards + g_idx * BOARD_SIZE;

    // find first empty cell
    int pos = -1;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (board[i] == 0) {
            pos = i;
            break;
        }
    }
    if (pos == -1) return;

    int r = pos / 9;
    int c = pos % 9;

    uint16_t mask = 0;

    // check row
    for (int i = 0; i < 9; i++) {
        int v = board[r * 9 + i];
        if (v) mask |= 1 << (v - 1);
    }

    // check column
    for (int i = 0; i < 9; i++) {
        int v = board[i * 9 + c];
        if (v) mask |= 1 << (v - 1);
    }

    // check box
    int br = (r / 3) * 3;
    int bc = (c / 3) * 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int v = board[(br + i) * 9 + (bc + j)];
            if (v) mask |= 1 << (v - 1);
        }
    }

    // generate children
    for (int d = 1; d <= 9; d++) {
        if (!(mask & (1 << (d - 1)))) {

            int idx = atomicAdd(newCount, 1);
            if (idx >= MAX_BFS_BOARDS) return;

            int* dst = newBoards + idx * BOARD_SIZE;

            int eidx = 0;
            for (int i = 0; i < BOARD_SIZE; i++) {
                dst[i] = board[i];
                if (board[i] == 0 && i != pos)
                    emptySpaces[idx * BOARD_SIZE + eidx++] = i;
            }

            dst[pos] = d;
            emptyCount[idx] = eidx;
        }
    }
}

void cuda_sudoku_backtrack(int* boards,
    int num_boards,
    int* empty_spaces,
    int* empty_count,
    int* finished,
    int* solution) {

    sudoku_backtrack << <NUM_BLOCKS, NUM_THREADS >> > (boards, num_boards, empty_spaces, empty_count, finished, solution);

}

void okej(int* old_boards,
    int* new_boards,
    int old_count,
    int* new_count,
    int* empty_spaces,
    int* empty_count) {

    sudoku_BFS << <NUM_BLOCKS, NUM_THREADS >> > (old_boards, new_boards, old_count, new_count, empty_spaces, empty_count);


}


