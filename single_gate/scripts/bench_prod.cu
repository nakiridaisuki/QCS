// One-gate benchmark using the production local_gate.cu kernels.
// Used by table C.0.3 for the "after" column; the "before" column still comes
// from the frozen legacy a197aaa bench_gate.
//
// Compile standalone:
//   nvcc -O3 -std=c++17 -arch=sm_90 bench_prod.cu -o bench_prod

#define main local_gate_unused_main
#include "local_gate.cu"
#undef main

#include <cstdio>
#include <functional>

template <typename T>
static double time_loop(int iters, const std::function<void()>& launch) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    launch();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

template <typename T>
static void run(int n, int iters) {
    using C = Complex<T>;
    const uint64_t size = 1ULL << n;
    const size_t bytes = size * sizeof(C);
    C* d = nullptr;
    cudaMalloc(&d, bytes);
    auto reset = [&] {
        cudaMemset(d, 0, bytes);
        C one{1, 0};
        cudaMemcpy(d, &one, sizeof(C), cudaMemcpyHostToDevice);
    };
    const T s = (T)(1.0 / std::sqrt(2.0));

    reset();
    for (int t : {0, 1, 2}) {
        reset();
        double us = time_loop<T>(iters, [&] {
            launch_u_gate<T>(d, n, t, {s,0}, {s,0}, {s,0}, {-s,0});
        }) * 1e3;
        std::printf("  u(t=%2d) %10.3f us/gate\n", t, us);
    }
    for (auto pr : {std::pair<int,int>{0,2}, {0,1}}) {
        reset();
        double us = time_loop<T>(iters, [&] { launch_swap_gate<T>(d, n, pr.first, pr.second); }) * 1e3;
        std::printf("  swap(%2d,%2d) %10.3f us/gate\n", pr.first, pr.second, us);
    }
    for (auto pr : {std::pair<int,int>{0,2}, {0,1}}) {
        reset();
        double us = time_loop<T>(iters, [&] { launch_cx_gate<T>(d, n, pr.first, pr.second); }) * 1e3;
        std::printf("  cx(%2d,%2d)   %10.3f us/gate\n", pr.first, pr.second, us);
    }
    for (auto pr : {std::pair<int,int>{0,12}, {0,2}}) {
        reset();
        double us = time_loop<T>(iters, [&] {
            launch_cp_gate<T>(d, n, pr.first, pr.second, {std::cos((T)0.7), std::sin((T)0.7)});
        }) * 1e3;
        std::printf("  cp(%2d,%2d)   %10.3f us/gate\n", pr.first, pr.second, us);
    }
    cudaFree(d);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <n> <fp32|fp64> [iters]\n", argv[0]);
        return 1;
    }
    int n = std::atoi(argv[1]);
    int iters = argc == 4 ? std::atoi(argv[3]) : 50;
    std::string p = argv[2];
    if (p == "fp32") run<float>(n, iters);
    else if (p == "fp64") run<double>(n, iters);
    else { std::fprintf(stderr, "precision must be fp32 or fp64\n"); return 1; }
    return 0;
}
