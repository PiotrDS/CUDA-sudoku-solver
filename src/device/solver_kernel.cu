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
    int num_boards,
    int* empty_spaces,
    int* empty_count,
    int* finished,
    int* solved
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_boards) return;
    if (atomicAdd(finished, 0)) return;

    int* board = boards + tid * SIZE;
    int* empties = empty_spaces + tid * SIZE;
    int total = empty_count[tid];

    int idx = 0;
    // fill empty cells
    while (idx >= 0 && idx < total && !atomicAdd(finished, 0)) {
        int pos = empties[idx];
        int& cell = board[pos];
        cell++;

        if (cell <= 9 && valid_board(board, pos)) {
            idx++;
        }
        else if (cell >= 9) {
            cell = 0;
            idx--;
        }
    }

    if (idx == total) {
        if (atomicCAS(finished, 0, 1) == 0) {
            for (int i = 0; i < SIZE; i++)
                solved[i] = board[i];
        }
    }
}


__global__
void sudoku_BFS(
    const int* old_boards,
    int* new_boards,
    int old_count,
    int* new_count,
    int* empty_spaces,
    int* empty_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= old_count) return;

    const int* board = old_boards + tid * SIZE;

    // find first empty
    int pos = -1;
    for (int i = 0; i < SIZE; i++) {
        if (board[i] == 0) {
            pos = i;
            break;
        }
    }
    if (pos < 0) return;

    // build constraint mask
    uint16_t mask = 0;
    int r = pos / 9;
    int c = pos % 9;

    for (int i = 0; i < 9; i++) {
        int x = board[r * 9 + i];
        if (x) mask |= 1 << (x - 1);
        x = board[i * 9 + c];
        if (x) mask |= 1 << (x - 1);
    }

    int br = (r / 3) * 3;
    int bc = (c / 3) * 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            int x = board[(br + i) * 9 + (bc + j)];
            if (x) mask |= 1 << (x - 1);
        }

    // generate boards
    for (int d = 1; d <= 9; d++) {
        if (mask & (1 << (d - 1))) continue;

        int idx = atomicAdd(new_count, 1);
        if (idx >= MAX_BFS_BOARDS) return;

        int* dst = new_boards + idx * SIZE;

        int eidx = 0;
        for (int i = 0; i < SIZE; i++) {
            dst[i] = board[i];
            if (board[i] == 0 && i != pos) {
                empty_spaces[idx * SIZE + eidx++] = i;
            }
        }

        dst[pos] = d;
        empty_count[idx] = eidx;
    }
}

void cuda_sudoku_backtrack(
    int num_blocks,
    int num_threads,
    int* boards,
    int num_boards,
    int* empty_spaces,
    int* empty_count,
    int* finished,
    int* solution) {

    // Record times 
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    float ms_total = 0.0f;

    // start total timer
    cudaEventRecord(start_total);

    sudoku_backtrack << <num_blocks, num_threads >> > (boards, num_boards, empty_spaces, empty_count, finished, solution);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    cudaEventElapsedTime(&ms_total, start_total, stop_total);

    //print results

    printf("\n===== GPU BACKTRACKING TIMING =====\n");
    printf("Total:                   %.3f ms\n", ms_total);
    printf("=========================\n\n");


    //clear events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);

}

float cuda_sudoku_BFS(
    int num_blocks,
    int num_threads,
    int* old_boards,
    int* new_boards,
    int old_count,
    int* new_count,
    int* empty_spaces,
    int* empty_count) {

    // Record times 
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    float ms_total = 0.0f;

    // start total timer
    cudaEventRecord(start_total);

    sudoku_BFS << <num_blocks, num_threads >> > (old_boards, new_boards, old_count, new_count, empty_spaces, empty_count);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    cudaEventElapsedTime(&ms_total, start_total, stop_total);


    //clear events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);

    return ms_total;
}


