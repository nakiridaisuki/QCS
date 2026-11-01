// DEPRECATED (appendix B reproduction only)
// Frozen copy of the pre-fix swap implementation from commit 18bd1f7
// ("Add CUDA-Q vs local gate study"), before a190b66 (lane kernel) and
// f95017b (wide/aligned Complex).  Do not include this in the normal build;
// single_gate/jobs/appB.*.sh compile it into single_gate/build/bench_gate_prelane.
// bench_gate.cu
//
// Per-gate microbenchmark: apply ONE gate to a state vector many times and
// report the average time, for our own local-gate kernels and for the four
// cuStateVec entry points.  This isolates the "cost of applying a gate" from
// circuit decomposition / fusion.
//
// Usage: bench_gate <n_qubits> <fp32|fp64> [iterations]

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define CSV_CHECK(x) do { auto s=(x); if(s!=CUSTATEVEC_STATUS_SUCCESS) throw std::runtime_error("cuStateVec status="+std::to_string((int)s)); } while(0)

template <typename T> struct Complex { T real; T imag; };

template <typename T>
struct Traits;
template <> struct Traits<float> {
    using C = cuFloatComplex;
    using HostComplex = Complex<float>;
    static constexpr cudaDataType_t dt = CUDA_C_32F;
    static constexpr custatevecComputeType_t ct = CUSTATEVEC_COMPUTE_32F;
    static C make(double r, double i) { return make_cuFloatComplex((float)r, (float)i); }
};
template <> struct Traits<double> {
    using C = cuDoubleComplex;
    using HostComplex = Complex<double>;
    static constexpr cudaDataType_t dt = CUDA_C_64F;
    static constexpr custatevecComputeType_t ct = CUSTATEVEC_COMPUTE_64F;
    static C make(double r, double i) { return make_cuDoubleComplex(r, i); }
};

// ---------------------------------------------------------------- local kernels

template <typename T>
__host__ __device__ inline Complex<T> cmul(Complex<T> a, Complex<T> b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

__device__ inline uint64_t insert_zero_bit(uint64_t compact, int qubit) {
    uint64_t low_mask = (1ULL << qubit) - 1ULL;
    uint64_t low = compact & low_mask;
    uint64_t high = compact & ~low_mask;
    return low | (high << 1);
}

__device__ inline uint64_t insert_two_zero_bits(uint64_t compact, int a, int b) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    return insert_zero_bit(insert_zero_bit(compact, lo), hi);
}

template <typename T>
__global__ void local_u_kernel(Complex<T>* state, uint64_t pairs, int target,
                               Complex<T> m00, Complex<T> m01,
                               Complex<T> m10, Complex<T> m11) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pairs) return;
    uint64_t i0 = insert_zero_bit(tid, target);
    uint64_t i1 = i0 | (1ULL << target);
    Complex<T> a0 = state[i0], a1 = state[i1];
    state[i0] = {m00.real*a0.real - m00.imag*a0.imag + m01.real*a1.real - m01.imag*a1.imag,
                 m00.real*a0.imag + m00.imag*a0.real + m01.real*a1.imag + m01.imag*a1.real};
    state[i1] = {m10.real*a0.real - m10.imag*a0.imag + m11.real*a1.real - m11.imag*a1.imag,
                 m10.real*a0.imag + m10.imag*a0.real + m11.real*a1.imag + m11.imag*a1.real};
}

template <typename T>
__global__ void local_cp_kernel(Complex<T>* state, uint64_t bases, int control,
                                int target, Complex<T> phase) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bases) return;
    uint64_t base = insert_two_zero_bits(tid, control, target);
    uint64_t i11 = base | (1ULL << control) | (1ULL << target);
    state[i11] = cmul(phase, state[i11]);
}

template <typename T>
__global__ void local_swap_kernel(Complex<T>* state, uint64_t bases, int a, int b) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bases) return;
    uint64_t base = insert_two_zero_bits(tid, a, b);
    uint64_t i01 = base | (1ULL << b);
    uint64_t i10 = base | (1ULL << a);
    Complex<T> t = state[i01];
    state[i01] = state[i10];
    state[i10] = t;
}

template <typename T>
__global__ void local_cx_kernel(Complex<T>* state, uint64_t bases, int control, int target) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bases) return;
    uint64_t base = insert_two_zero_bits(tid, control, target);
    uint64_t i0 = base | (1ULL << control);
    uint64_t i1 = i0 | (1ULL << target);
    Complex<T> t = state[i0];
    state[i0] = state[i1];
    state[i1] = t;
}

// ---------------------------------------------------------------- runner

template <typename T>
struct TimeResult { double ms_per_gate; double norm; };

class CsvHandle {
public:
    custatevecHandle_t handle{};
    CsvHandle() { CSV_CHECK(custatevecCreate(&handle)); }
    ~CsvHandle() { if (handle) custatevecDestroy(handle); }
    CsvHandle(const CsvHandle&) = delete;
    CsvHandle& operator=(const CsvHandle&) = delete;
};

template <typename T> void run(int n, int iters);

int main(int argc, char** argv) {
    try {
        if (argc < 3) { std::fprintf(stderr, "Usage: %s <n_qubits> <fp32|fp64> [iterations]\n", argv[0]); return 1; }
        int n = std::atoi(argv[1]);
        std::string prec = argv[2];
        int iters = argc > 3 ? std::atoi(argv[3]) : (n >= 25 ? 20 : 100);
        if (n < 2 || n >= 31) throw std::runtime_error("n_qubits must be 2..30");

        if (prec == "fp32") { run<float>(n, iters); return 0; }
        if (prec == "fp64") { run<double>(n, iters); return 0; }
        throw std::runtime_error("precision must be fp32 or fp64");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}

template <typename T>
double time_kernel_loop(int iters, std::function<void()> launch) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    launch();  // warm-up
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return (double)ms / iters;
}

template <typename T>
void run(int n, int iters) {
    using C = typename Traits<T>::C;
    using HC = typename Traits<T>::HostComplex;
    using TR = Traits<T>;

    uint64_t size = 1ULL << n;
    size_t bytes = size * sizeof(C);
    C* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));

    std::printf("n=%d  precision=%s  iters=%d\n", n, sizeof(T) == 4 ? "fp32" : "fp64", iters);

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(d, 0, bytes));
        C one = TR::make(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d, &one, sizeof(C), cudaMemcpyHostToDevice));
    };
    auto norm = [&]() {
        std::vector<C> h(size);
        CUDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
        double s = 0; for (auto& z : h) s += (double)z.x * z.x + (double)z.y * z.y;
        return s;
    };
    auto show = [&](const char* name, double ms) {
        std::printf("  %-22s %10.3f us/gate\n", name, ms * 1e3);
    };

    const int tgt = n - 1, ctrl = 0, q0 = 0, q1 = 1;
    uint64_t pairs = 1ULL << (n - 1), bases = 1ULL << (n - 2);
    constexpr int TH = 256;
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    Complex<T> lu00{ (T)inv_sqrt2, 0 }, lu01{ (T)inv_sqrt2, 0 };
    Complex<T> lu10{ (T)inv_sqrt2, 0 }, lu11{ (T)-inv_sqrt2, 0 };
    Complex<T> phase{ (T)std::cos(0.7), (T)std::sin(0.7) };

    // local u
    reset();
    show("local_u (generic)", time_kernel_loop<T>(iters, [&]{
        local_u_kernel<T><<<(pairs + TH - 1) / TH, TH>>>(
            reinterpret_cast<Complex<T>*>(d), pairs, tgt, lu00, lu01, lu10, lu11);
    }));
    std::printf("      norm(local_u)=%.6f\n", norm());

    // local cp
    reset();
    show("local_cp (ctrl diag)", time_kernel_loop<T>(iters, [&]{
        local_cp_kernel<T><<<(bases + TH - 1) / TH, TH>>>(
            reinterpret_cast<Complex<T>*>(d), bases, ctrl, tgt, phase);
    }));

    // local swap
    reset();
    show("local_swap (perm)", time_kernel_loop<T>(iters, [&]{
        local_swap_kernel<T><<<(bases + TH - 1) / TH, TH>>>(reinterpret_cast<Complex<T>*>(d), bases, q0, q1);
    }));

    // local cx (same structure as swap, different bit pair)
    reset();
    show("local_cx (perm)", time_kernel_loop<T>(iters, [&]{
        local_cx_kernel<T><<<(bases + TH - 1) / TH, TH>>>(reinterpret_cast<Complex<T>*>(d), bases, ctrl, tgt);
    }));

    // cuStateVec variants
    CsvHandle h;
    void* ws = nullptr; size_t ws_cap = 0;
    auto ensure_ws = [&](size_t need) {
        if (need > ws_cap) { if (ws) cudaFree(ws); CUDA_CHECK(cudaMalloc(&ws, need)); ws_cap = need; }
    };
    auto csv_matrix = [&](C* mat, int target, const int32_t* ctrls, uint32_t nctrls) {
        int32_t targets[] = { target };
        size_t need = 0;
        CSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(h.handle, TR::dt, n, mat, TR::dt,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, nctrls, TR::ct, &need));
        ensure_ws(need);
        CSV_CHECK(custatevecApplyMatrix(h.handle, d, TR::dt, n, mat, TR::dt,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, ctrls, nullptr, nctrls, TR::ct, ws, need));
    };

    C hmat[4] = { TR::make(inv_sqrt2,0), TR::make(inv_sqrt2,0), TR::make(inv_sqrt2,0), TR::make(-inv_sqrt2,0) };
    reset();
    show("csv_ApplyMatrix u", time_kernel_loop<T>(iters, [&]{ csv_matrix(hmat, tgt, nullptr, 0); }));

    C xmat[4] = { TR::make(0,0), TR::make(1,0), TR::make(1,0), TR::make(0,0) };
    int32_t ctrls[] = { ctrl };
    reset();
    show("csv_ApplyMatrix cx", time_kernel_loop<T>(iters, [&]{ csv_matrix(xmat, tgt, ctrls, 1); }));

    C dmat[4] = { TR::make(1,0), TR::make(0,0), TR::make(0,0), TR::make(std::cos(0.7), std::sin(0.7)) };
    reset();
    show("csv_ApplyMatrix cp(diag)", time_kernel_loop<T>(iters, [&]{ csv_matrix(dmat, tgt, ctrls, 1); }));

    // cp via generalized permutation with identity permutation + diagonal
    {
        int32_t targets[] = { tgt };
        custatevecIndex_t perm[2] = { 0, 1 };
        C pdiag[2] = { TR::make(1,0), TR::make(std::cos(0.7), std::sin(0.7)) };
        size_t need = 0;
        CSV_CHECK(custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(h.handle, TR::dt, n, perm, pdiag, TR::dt, targets, 1, 1, &need));
        ensure_ws(need);
        reset();
        show("csv_Permutation cp(diag)", time_kernel_loop<T>(iters, [&]{
            CSV_CHECK(custatevecApplyGeneralizedPermutationMatrix(h.handle, d, TR::dt, n, perm, pdiag, TR::dt, 0,
                targets, 1, ctrls, nullptr, 1, ws, need));
        }));
    }

    // Pauli rotation (rz): 1 target, no control
    reset();
    show("csv_ApplyPauliRotation", time_kernel_loop<T>(iters, [&]{
        custatevecPauli_t paulis[] = { CUSTATEVEC_PAULI_Z };
        int32_t targets[] = { tgt };
        CSV_CHECK(custatevecApplyPauliRotation(h.handle, d, TR::dt, n, 0.7, paulis, targets, 1, nullptr, nullptr, 0));
    }));

    // Generalized permutation for CX: target t, control c
    {
        int32_t targets[] = { tgt };
        custatevecIndex_t perm[2] = { 1, 0 };
        C diag[2] = { TR::make(1,0), TR::make(1,0) };
        size_t need = 0;
        CSV_CHECK(custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(h.handle, TR::dt, n, perm, diag, TR::dt, targets, 1, 1, &need));
        ensure_ws(need);
        reset();
        show("csv_Permutation cx", time_kernel_loop<T>(iters, [&]{
            CSV_CHECK(custatevecApplyGeneralizedPermutationMatrix(h.handle, d, TR::dt, n, perm, diag, TR::dt, 0,
                targets, 1, ctrls, nullptr, 1, ws, need));
        }));
    }

    // Generalized permutation for swap: targets {q0,q1}
    {
        int32_t targets[] = { q0, q1 };
        custatevecIndex_t perm[4] = { 0, 2, 1, 3 };
        C diag[4] = { TR::make(1,0), TR::make(1,0), TR::make(1,0), TR::make(1,0) };
        size_t need = 0;
        CSV_CHECK(custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(h.handle, TR::dt, n, perm, diag, TR::dt, targets, 2, 0, &need));
        ensure_ws(need);
        reset();
        show("csv_Permutation swap", time_kernel_loop<T>(iters, [&]{
            CSV_CHECK(custatevecApplyGeneralizedPermutationMatrix(h.handle, d, TR::dt, n, perm, diag, TR::dt, 0,
                targets, 2, nullptr, nullptr, 0, ws, need));
        }));
        // correctness: |q0=1,q1=0> (index 1) must move to |0,1> (index 2)
        CUDA_CHECK(cudaMemset(d, 0, bytes));
        C one = TR::make(1,0);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<C*>(d) + 1, &one, sizeof(C), cudaMemcpyHostToDevice));
        CSV_CHECK(custatevecApplyGeneralizedPermutationMatrix(h.handle, d, TR::dt, n, perm, diag, TR::dt, 0,
            targets, 2, nullptr, nullptr, 0, ws, need));
        std::vector<C> hh(4);
        CUDA_CHECK(cudaMemcpy(hh.data(), d, 4 * sizeof(C), cudaMemcpyDeviceToHost));
        std::printf("      swap check: |01>->%d  |10>->%d  (expect 2 and 1)\n",
                    hh[2].x > 0.5 ? 2 : 0, hh[1].x > 0.5 ? 1 : 0);
    }

    // D2H / H2D transfer cost for the full state vector (time only, no compute)
    {
        std::vector<C> host(size);
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        const int reps = 5;
        CUDA_CHECK(cudaEventRecord(a));
        for (int i = 0; i < reps; ++i)
            CUDA_CHECK(cudaMemcpy(host.data(), d, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        double per = ms / reps;
        std::printf("  %-22s %10.3f ms  (%.1f GB/s, %.1f MB)\n", "D2H memcpy", per,
                    bytes / 1e9 / (per / 1e3), bytes / 1e6);
        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    }

    if (ws) CUDA_CHECK(cudaFree(ws));
    CUDA_CHECK(cudaFree(d));
}