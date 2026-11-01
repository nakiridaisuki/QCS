// local_gate.cu
//
// Full State Vector Simulation
// IR source (qiskit_qasm_parser.py):
//   Qiskit -> basis_gates=["u", "cx"]                 (u/cx IR)
//   Qiskit -> basis_gates=["u", "cx", "cp", "swap"]  (native IR, --native)
//
// IR format (whitespace-separated, one gate per line):
//   u    <target> <u00.re u00.im u01.re u01.im u10.re u10.im u11.re u11.im>
//   cx   <control> <target>
//   cp   <control> <target> <re> <im>   (phase = e^{i*lambda})
//   swap <a> <b>

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

// Complex number
//
// alignas(2*sizeof(T)) matters for performance, not correctness. Without it the
// compiler emits two separate 8 B LDG.E.64 / 4 B LDG.E per complex (real and
// imag apart) and, when the operated bit position is low, consecutive threads
// end up one-per-32 B-sector. Measured on H200 (fp64, n=25, applying an arbitrary
// 1-qubit gate to target bit 0 vs bit 24): 419 us vs 257 us. ncu shows DRAM and
// L2 traffic are identical in both cases - only the L1 T-stage sector-request
// volume doubles (32 vs 16 sectors per warp request). With the alignment below
// the SASS becomes 2 x LDG.E.128 (fp64) / 2 x LDG.E.64 (fp32), halving the
// requests, and the low-target penalty drops to 287 us (wide variant, job 612363).
// None of the other kernels need code changes: they all go through this struct.
//
// Alignment is guaranteed for the state vector because
//   - cudaMalloc returns 256 B aligned memory, and
//   - every access is at element granularity (index * sizeof(Complex<T>)),
// so each element is naturally 8 B (fp32) / 16 B (fp64) aligned.
// Only a raw byte-offset view of the state would break this.
template <typename T>
struct alignas(2 * sizeof(T)) Complex {
    T real;
    T imag;
};

// Gate

enum class GateType {
    U,
    CX,
    CP,
    SWAP
};

template <typename T>
struct Gate {
    GateType type;

    int target = -1;
    int control = -1;
    int target2 = -1;      // second qubit for SWAP

    Complex<T> u00{};
    Complex<T> u01{};
    Complex<T> u10{};
    Complex<T> u11{};
    Complex<T> phase{};    // e^{i*lambda} for CP
};

template <typename T>
__host__ __device__
Complex<T> complex_add(
    const Complex<T>& a,
    const Complex<T>& b)
{
    return {a.real + b.real, a.imag + b.imag};
}


template <typename T>
__host__ __device__
Complex<T> complex_mul(
    const Complex<T>& a,
    const Complex<T>& b)
{
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}


// ============================================================
// Index mapping
//
// Each thread handles ONE pair of amplitudes:
//
// target = 0:
//
// thread 0 -> state[0000], state[0001] 
// thread 1 -> state[0010], state[0011]
// ...
// target = 2:
//
// thread 0 -> state[0000], state[0100]
// thread 1 -> state[0001], state[0101]
// ... and so on
// ============================================================

__device__
uint64_t insert_zero_bit(
    uint64_t compact_index,
    int target_qubit)
{       
    
    uint64_t low_mask = (1ULL << target_qubit) - 1ULL;

    // take the lower bits
    uint64_t low  = compact_index & low_mask;
    // take the higher bits
    uint64_t high = compact_index & ~low_mask;

    return low | (high << 1);
}


// ============================================================
// Single-qubit U gate kernel
//
// state:
//   Full 2^n state vector in GPU global memory
//
// num_pairs:
//   2^(n-1)
//
// target:
//   target qubit index
//
// u00 ~ u11:
//   2x2 quantum gate matrix (not 2^n x 2^n matrix)
//
// One CUDA thread:
//   reads two amplitudes
//   performs one 2x2 matrix-vector multiplication
//   writes two amplitudes back
// ============================================================

template <typename T>
__global__
void apply_u_gate_kernel( // only for u gate
    Complex<T>* state,
    uint64_t num_pairs,
    int target,

    Complex<T> u00,
    Complex<T> u01,
    Complex<T> u10,
    Complex<T> u11)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (tid >= num_pairs)
        return;
    // the two state vector indices handled by this thread.
    // index0: target qubit = 0
    // index1: target qubit = 1

    const uint64_t index0 = insert_zero_bit(tid, target);
    const uint64_t index1 = index0 | (1ULL << target);

    // load amplitudes from global memory
    const Complex<T> a0 = state[index0];
    const Complex<T> a1 = state[index1];


    const Complex<T> b0 =
        complex_add(complex_mul(u00, a0), complex_mul(u01, a1));
    const Complex<T> b1 =
        complex_add(complex_mul(u10, a0), complex_mul(u11, a1));

    state[index0] = b0;
    state[index1] = b1;
}


// same concept as u gate, but we need to insert two zero bits for CX gate
// including target and control qubits

// ============================================================
// Warp-transpose U kernel (only used when target < LOGW)
//
// Same gate, but the work is rearranged so that every load and every store is
// fully contiguous, which removes the sector scatter that the plain kernel
// still has at low targets even after the Complex alignment fix.
//
// Layout: warp w owns the 2^(LOGW+1) CONSECUTIVE amplitudes
//   [w * 2^(LOGW+1), (w+1) * 2^(LOGW+1)).
// Lane l holds slot 0 = base + l and slot 1 = base + 2^LOGW + l. Since LOGW is 5,
// flipping a bit below LOGW maps an element to the SAME slot of lane
// (l ^ 2^target), so the partner comes from __shfl_sync instead of a second
// global load.
//
// CAVEATS - every one of these is load-bearing:
//   * Only valid for `target < LOGW`. For target >= LOGW the partner lives in a
//     different warp, so launch_u_gate falls back to the plain kernel.
//   * __shfl_sync(0xffffffff, ...) requires ALL 32 lanes of the warp to be
//     active and converged. The early return below is therefore a WARP-uniform
//     test on `warp`; do NOT rewrite it as a per-thread test on `tid`, that
//     would disable individual lanes and silently shuffle stale values.
//   * blockDim.x must be a multiple of 2^LOGW (one warp), and num_qubits must be
//     >= LOGW + 1 so that a full warp block exists.
//   * The grid packs (blockDim.x / 2^LOGW) warps per block.
// ============================================================

template <typename T>
__device__
inline Complex<T> complex_shfl(Complex<T> value, int src_lane)
{
    value.real = __shfl_sync(0xffffffffu, value.real, src_lane);
    value.imag = __shfl_sync(0xffffffffu, value.imag, src_lane);
    return value;
}

template <typename T, int LOGW>
__global__
void apply_u_gate_shfl_kernel(
    Complex<T>* state,
    uint64_t num_warps,
    int target,

    Complex<T> u00,
    Complex<T> u01,
    Complex<T> u10,
    Complex<T> u11)
{
    constexpr int LANES = 1 << LOGW;

    const uint64_t warp =
        static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> LOGW)
        + (threadIdx.x >> LOGW);

    // warp-uniform guard, see the caveat above
    if (warp >= num_warps)
        return;

    const int lane = threadIdx.x & (LANES - 1);
    const uint64_t base = warp << (LOGW + 1);

    const Complex<T> a0 = state[base + lane];
    const Complex<T> a1 = state[base + LANES + lane];

    const int partner_lane = lane ^ (1 << target);
    const Complex<T> p0 = complex_shfl(a0, partner_lane);
    const Complex<T> p1 = complex_shfl(a1, partner_lane);

    Complex<T> b0;
    Complex<T> b1;

    if ((lane >> target) & 1) {
        // this lane holds the |target=1> element of its pairs
        b0 = complex_add(complex_mul(u10, p0), complex_mul(u11, a0));
        b1 = complex_add(complex_mul(u10, p1), complex_mul(u11, a1));
    }
    else {
        b0 = complex_add(complex_mul(u00, a0), complex_mul(u01, p0));
        b1 = complex_add(complex_mul(u00, a1), complex_mul(u01, p1));
    }

    state[base + lane] = b0;
    state[base + LANES + lane] = b1;
}

__device__
uint64_t insert_two_zero_bits(
    uint64_t compact_index,
    int bit_a,
    int bit_b)
{
    const int low_bit  = (bit_a < bit_b) ? bit_a : bit_b;
    const int high_bit = (bit_a < bit_b) ? bit_b : bit_a;

    // Insert lower-position zero first.
    uint64_t index = insert_zero_bit(compact_index, low_bit);

    // Then insert the higher-position zero.
    index = insert_zero_bit(index, high_bit);

    return index;
}


// ============================================================
// CX kernel
// IR: cx <control> <target>
//
// CX behavior:
// control = 0:
//     state unchanged
// control = 1:
//     target=0 <-> target=1
//
// One CUDA thread handles one amplitude pair.
// ============================================================

template <typename T>
__global__
void apply_cx_gate_kernel(
    Complex<T>* state,
    uint64_t num_pairs,
    int control,
    int target)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (tid >= num_pairs)
        return;

    uint64_t base = insert_two_zero_bits(tid,control,target);

    // CX only acts when control = 1.
    // Set control bit to 1.
    // target remains 0.

    const uint64_t index0 = base | (1ULL << control); // control = 1, target = 0
    // Same state, but target = 1.
    const uint64_t index1 = index0 | (1ULL << target); // control = 1, target = 1

    // simply a swap between these two amplitudes.
    const Complex<T> temp = state[index0];

    state[index0] = state[index1];

    state[index1] = temp;
}

// ============================================================
// Controlled-phase kernel (cp)
// IR: cp <control> <target> <phase.real> <phase.imag>
//
// cp(lambda) = diag(1, 1, 1, e^{i*lambda}); only the
// |control=1, target=1> amplitude is scaled. One thread handles one
// (control=0, target=0) base index.
// ============================================================

template <typename T>
__global__
void apply_cp_gate_kernel(
    Complex<T>* state,
    uint64_t num_bases,
    int control,
    int target,
    Complex<T> phase)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= num_bases)
        return;

    const uint64_t base = insert_two_zero_bits(tid, control, target);
    const uint64_t index11 =
        base | (1ULL << control) | (1ULL << target);

    state[index11] = complex_mul(phase, state[index11]);
}

// ============================================================
// Vectorized swap (intra-chunk case)
//
// A 32-byte chunk holds VEC = 4 complexes for fp32 and 2 for fp64, so the
// lowest LOGB index bits (LOGB = 2 / 1) are "lane" bits inside one chunk.
//
// When BOTH swapped bits are lane bits, the whole permutation stays inside
// one chunk. At that point the scalar kernel below accesses only every other
// complex (stride 2^small), which costs ~1.25x over the DRAM roofline; here we
// load the chunk with one coalesced 32 B access, permute lanes in registers,
// and store it back.
//
// shortcut: any other bit pair keeps the scalar kernel -- measured at or below
// the 2x-state traffic floor already (see bench_gate sweep).
// ============================================================

template <typename T, int LOGB>
struct alignas(32) VecChunk {
    Complex<T> v[1 << LOGB];
};

template <typename T, int LOGB>
__global__
void apply_swap_gate_lane_kernel(
    Complex<T>* state,
    uint64_t num_chunks,
    int qubit_a,
    int qubit_b,
    int lane_mask)
{
    const uint64_t chunk =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (chunk >= num_chunks)
        return;

    using V = VecChunk<T, LOGB>;
    const V in = *reinterpret_cast<const V*>(state + (chunk << LOGB));

    V out;
    #pragma unroll
    for (int lane = 0; lane < (1 << LOGB); ++lane) {
        const int bit_a = (lane >> qubit_a) & 1;
        const int bit_b = (lane >> qubit_b) & 1;
        // swap only exchanges |a=0,b=1> with |a=1,b=0>
        out.v[lane] = (bit_a != bit_b)
            ? in.v[lane ^ lane_mask]
            : in.v[lane];
    }

    *reinterpret_cast<V*>(state + (chunk << LOGB)) = out;
}

// ============================================================
// Swap kernel
// IR: swap <qubit_a> <qubit_b>
//
// Exchanges the amplitudes with (a=0, b=1) and (a=1, b=0).
// One thread handles one (a=0, b=0) base index.
// ============================================================

template <typename T>
__global__
void apply_swap_gate_kernel(
    Complex<T>* state,
    uint64_t num_bases,
    int qubit_a,
    int qubit_b)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= num_bases)
        return;

    const uint64_t base = insert_two_zero_bits(tid, qubit_a, qubit_b);
    const uint64_t index01 = base | (1ULL << qubit_b);
    const uint64_t index10 = base | (1ULL << qubit_a);

    const Complex<T> temp = state[index01];
    state[index01] = state[index10];
    state[index10] = temp;
}

// Host-side launcher

template <typename T>
void launch_u_gate(
    Complex<T>* d_state,
    int num_qubits,
    int target,

    Complex<T> u00,
    Complex<T> u01,
    Complex<T> u10,
    Complex<T> u11,
    cudaStream_t stream = nullptr)
{
    if (num_qubits <= 0 || num_qubits >= 64)
        throw std::runtime_error("num_qubits must be between 1 and 63");

    if (target < 0 || target >= num_qubits)
        throw std::runtime_error("U target qubit is out of range");

    // Total state size:
    //
    // N = 2^num_qubits

    const uint64_t state_size = 1ULL << num_qubits;

    // need 2^(n-1) threads
    const uint64_t num_pairs = state_size >> 1;


    // Initial configuration.

    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int LOGW = 5;   // lane bits used by the warp-transpose kernel

    // Low target bits: the plain kernel spreads each warp request over twice the
    // necessary number of L1 sectors (see the comment on Complex above).
    // target < LOGW means the pair stays inside a single warp, so the transpose
    // kernel can make every access contiguous instead.
    if (target < LOGW && num_qubits >= LOGW + 1) {
        const uint64_t num_warps = state_size >> (LOGW + 1);
        const uint64_t warps_per_block = THREADS_PER_BLOCK >> LOGW;
        const uint64_t shfl_blocks =
            (num_warps + warps_per_block - 1) / warps_per_block;

        apply_u_gate_shfl_kernel<T,LOGW>
            <<<shfl_blocks,THREADS_PER_BLOCK,0,stream>>>(
                d_state,
                num_warps,
                target,
                u00,
                u01,
                u10,
                u11
            );

        const cudaError_t shfl_status = cudaGetLastError();
        if (shfl_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("apply_u_gate_shfl_kernel launch failed: ") +
                cudaGetErrorString(shfl_status));
        }
        return;
    }

    // NOTE: unlike the u gate, CX deliberately has no warp-transpose variant.
    // The transpose kernel has to touch the whole state, while the pair kernel
    // above touches only the control=1 half. Measured on H200 (jobs 612550 and
    // 612563, after the Complex alignment fix so both sides are the shipped
    // code): fp64 n=23, control=2/target=3 is 49.6 us with this kernel and
    // 68.2 us transposed; fp32 n=25, control=2/target=3 is 120.2 vs 128.2 us.
    // The transpose only wins when the control bit sits below the sector size,
    // i.e. fp64 control 0 (71.2 -> 68.6 us) or fp32 control 0 (143.6 -> 130.8),
    // which is a 4-9% gain on a subset. Not worth a second kernel + dispatch;
    // the win here comes from the wide (aligned) access in Complex<T>.

    const uint64_t num_blocks = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    apply_u_gate_kernel<T>
        <<<num_blocks,THREADS_PER_BLOCK,0,stream>>>(
            d_state,
            num_pairs,
            target,
            u00,
            u01,
            u10,
            u11
        );
    


    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("apply_u_gate_kernel launch failed: ") +
            cudaGetErrorString(launch_status));
    }
}

template <typename T>
void launch_cx_gate(
    Complex<T>* d_state,
    int num_qubits,
    int control,
    int target,
    cudaStream_t stream = nullptr)
{
    if (num_qubits < 2 || num_qubits >= 64)
        throw std::runtime_error("CX requires 2 to 63 qubits");

    if (control < 0 || control >= num_qubits ||
        target < 0 || target >= num_qubits)
        throw std::runtime_error("CX qubit index is out of range");

    if (control == target)
        throw std::runtime_error("CX control and target cannot be the same qubit");


    // For CX: number of thread pairs = 2^(n-2)

    const uint64_t num_pairs = 1ULL << (num_qubits - 2);
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int LOGW = 5;

    const uint64_t num_blocks = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    apply_cx_gate_kernel<T>
        <<<num_blocks,THREADS_PER_BLOCK,0,stream>>>(
            d_state,
            num_pairs,
            control,
            target
        );


    const cudaError_t launch_status = cudaGetLastError();

    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("apply_cx_gate_kernel launch failed: ") +
            cudaGetErrorString(launch_status)
        );
    }
}

template <typename T>
void launch_cp_gate(
    Complex<T>* d_state,
    int num_qubits,
    int control,
    int target,
    Complex<T> phase,
    cudaStream_t stream = nullptr)
{
    if (num_qubits < 2 || num_qubits >= 64)
        throw std::runtime_error("cp requires 2 to 63 qubits");

    if (control < 0 || control >= num_qubits ||
        target < 0 || target >= num_qubits)
        throw std::runtime_error("cp qubit index is out of range");

    if (control == target)
        throw std::runtime_error("cp control and target cannot be the same qubit");

    const uint64_t num_bases = 1ULL << (num_qubits - 2);
    constexpr int THREADS_PER_BLOCK = 256;
    const uint64_t num_blocks = (num_bases + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    apply_cp_gate_kernel<T>
        <<<num_blocks,THREADS_PER_BLOCK,0,stream>>>(
            d_state,
            num_bases,
            control,
            target,
            phase
        );

    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("apply_cp_gate_kernel launch failed: ") +
            cudaGetErrorString(launch_status));
    }
}

template <typename T>
void launch_swap_gate(
    Complex<T>* d_state,
    int num_qubits,
    int qubit_a,
    int qubit_b,
    cudaStream_t stream = nullptr)
{
    if (num_qubits < 2 || num_qubits >= 64)
        throw std::runtime_error("swap requires 2 to 63 qubits");

    if (qubit_a < 0 || qubit_a >= num_qubits ||
        qubit_b < 0 || qubit_b >= num_qubits)
        throw std::runtime_error("swap qubit index is out of range");

    if (qubit_a == qubit_b)
        throw std::runtime_error("swap qubits must be distinct");

    const uint64_t num_bases = 1ULL << (num_qubits - 2);
    constexpr int THREADS_PER_BLOCK = 256;

    // Both bits inside one 32-byte chunk -> intra-chunk permutation (coalesced).
    constexpr int LOGB = (sizeof(T) == 4) ? 2 : 1;
    if (qubit_a < LOGB && qubit_b < LOGB) {
        const uint64_t num_chunks = 1ULL << (num_qubits - LOGB);
        const uint64_t num_blocks =
            (num_chunks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const int lane_mask = (1 << qubit_a) | (1 << qubit_b);

        apply_swap_gate_lane_kernel<T,LOGB>
            <<<num_blocks,THREADS_PER_BLOCK,0,stream>>>(
                d_state,
                num_chunks,
                qubit_a,
                qubit_b,
                lane_mask
            );

        const cudaError_t lane_status = cudaGetLastError();
        if (lane_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("apply_swap_gate_lane_kernel launch failed: ") +
                cudaGetErrorString(lane_status));
        }
        return;
    }

    const uint64_t num_blocks = (num_bases + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    apply_swap_gate_kernel<T>
        <<<num_blocks,THREADS_PER_BLOCK,0,stream>>>(
            d_state,
            num_bases,
            qubit_a,
            qubit_b
        );

    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("apply_swap_gate_kernel launch failed: ") +
            cudaGetErrorString(launch_status));
    }
}

template <typename T>
void execute_circuit(
    Complex<T>* d_state,
    int num_qubits,
    const std::vector<Gate<T>>& gates,
    cudaStream_t stream = nullptr)
{
    for (const auto& gate : gates) {
        if (gate.type == GateType::U) {
            launch_u_gate<T>(
                d_state,
                num_qubits,
                gate.target,
                gate.u00,
                gate.u01,
                gate.u10,
                gate.u11,
                stream
            );
        }
        else if (gate.type == GateType::CX) {
            launch_cx_gate<T>(
                d_state,
                num_qubits,
                gate.control,
                gate.target,
                stream
            );
        }
        else if (gate.type == GateType::CP) {
            launch_cp_gate<T>(
                d_state,
                num_qubits,
                gate.control,
                gate.target,
                gate.phase,
                stream
            );
        }
        else if (gate.type == GateType::SWAP) {
            launch_swap_gate<T>(
                d_state,
                num_qubits,
                gate.target,
                gate.target2,
                stream
            );
        }
    }

}



template <typename T>
int run(const std::string& circuit_file, const std::string& output_file)
{
    const auto load_start = std::chrono::high_resolution_clock::now();
    std::ifstream file(circuit_file);
    if (!file)
        throw std::runtime_error("Cannot open circuit file: " + circuit_file);

    std::string keyword;

    // ------------------------------------------------------------
    // Read IR header
    // ------------------------------------------------------------

    file >> keyword;
    if (keyword != "QISKIT_CIRCUIT_V1")
        throw std::runtime_error("Invalid circuit IR format");

    int num_qubits;
    file >> keyword >> num_qubits;
    if (keyword != "qubits")
        throw std::runtime_error("Expected 'qubits'");
    if (num_qubits <= 0 || num_qubits >= 64)
        throw std::runtime_error("num_qubits must be between 1 and 63");

    double phase_real;
    double phase_imag;
    file >> keyword >> phase_real >> phase_imag;
    if (keyword != "global_phase")
        throw std::runtime_error("Expected 'global_phase'");

    size_t num_gates;
    file >> keyword >> num_gates;
    if (keyword != "gates")
        throw std::runtime_error("Expected 'gates'");

    // ------------------------------------------------------------
    // Read gates
    // ------------------------------------------------------------

    std::vector<Gate<T>> gates;
    gates.reserve(num_gates);

    for (size_t i = 0; i < num_gates; ++i) {
        std::string gate_name;
        file >> gate_name;

        if (gate_name == "u") {
            Gate<T> gate;
            gate.type = GateType::U;

            double u00r, u00i, u01r, u01i;
            double u10r, u10i, u11r, u11i;

            file >> gate.target
                 >> u00r >> u00i
                 >> u01r >> u01i
                 >> u10r >> u10i
                 >> u11r >> u11i;

            gate.u00 = {static_cast<T>(u00r), static_cast<T>(u00i)};
            gate.u01 = {static_cast<T>(u01r), static_cast<T>(u01i)};
            gate.u10 = {static_cast<T>(u10r), static_cast<T>(u10i)};
            gate.u11 = {static_cast<T>(u11r), static_cast<T>(u11i)};

            gates.push_back(gate);
        }
        else if (gate_name == "cx") {
            Gate<T> gate;
            gate.type = GateType::CX;

            file >> gate.control >> gate.target;

            gates.push_back(gate);
        }
        else if (gate_name == "cp") {
            Gate<T> gate;
            gate.type = GateType::CP;

            double phase_real, phase_imag;
            file >> gate.control >> gate.target >> phase_real >> phase_imag;

            gate.phase = {static_cast<T>(phase_real), static_cast<T>(phase_imag)};

            gates.push_back(gate);
        }
        else if (gate_name == "swap") {
            Gate<T> gate;
            gate.type = GateType::SWAP;

            file >> gate.target >> gate.target2;

            gates.push_back(gate);
        }
        else {
            throw std::runtime_error("Unknown gate: " + gate_name);
        }
    }

    const auto load_end = std::chrono::high_resolution_clock::now();
    const double load_time_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();

    // ------------------------------------------------------------
    // Allocate state vector
    // ------------------------------------------------------------

    const uint64_t state_size = 1ULL << num_qubits;
    const size_t state_bytes = state_size * sizeof(Complex<T>);

    Complex<T>* d_state = nullptr;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_state), state_bytes);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));

    status = cudaMemset(d_state, 0, state_bytes);
    if (status != cudaSuccess) {
        cudaFree(d_state);
        throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(status));
    }

    // ------------------------------------------------------------
    // Initial state
    //
    // Normally |00...0> has state[0] = 1.
    //
    // The Qiskit IR also contains global_phase = exp(i*phi).
    // Since global phase commutes with the entire circuit, initialize:
    //
    // state[0] = global_phase
    //
    // instead of applying another kernel at the end.
    // ------------------------------------------------------------

    const Complex<T> initial_amplitude = {
        static_cast<T>(phase_real),
        static_cast<T>(phase_imag)
    };

    status = cudaMemcpy(d_state, &initial_amplitude, sizeof(Complex<T>), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(d_state);
        throw std::runtime_error(std::string("Initial state copy failed: ") + cudaGetErrorString(status));
    }

    // ------------------------------------------------------------
    // Execute circuit
    // ------------------------------------------------------------

    auto reset_state = [&]() -> cudaError_t {
        cudaError_t rc = cudaMemset(d_state, 0, state_bytes);
        if (rc != cudaSuccess) return rc;
        return cudaMemcpy(d_state, &initial_amplitude, sizeof(Complex<T>), cudaMemcpyHostToDevice);
    };

    cudaEvent_t simulation_start;
    cudaEvent_t simulation_end;

    status = cudaEventCreate(&simulation_start);
    if (status != cudaSuccess) {
        cudaFree(d_state);
        throw std::runtime_error(std::string("Start event creation failed: ") + cudaGetErrorString(status));
    }

    status = cudaEventCreate(&simulation_end);
    if (status != cudaSuccess) {
        cudaEventDestroy(simulation_start);
        cudaFree(d_state);
        throw std::runtime_error(std::string("End event creation failed: ") + cudaGetErrorString(status));
    }

    constexpr int warmup_runs = 1;
    constexpr int timed_runs = 3;
    float simulation_time_ms = 0.0F;

    try {
        for (int i = 0; i < warmup_runs; ++i) {
            status = reset_state();
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            execute_circuit<T>(d_state, num_qubits, gates);
        }
        for (int i = 0; i < timed_runs; ++i) {
            status = reset_state();
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            status = cudaEventRecord(simulation_start);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            execute_circuit<T>(d_state, num_qubits, gates);
            status = cudaEventRecord(simulation_end);
            if (status == cudaSuccess) status = cudaEventSynchronize(simulation_end);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            float one_run = 0.0F;
            status = cudaEventElapsedTime(&one_run, simulation_start, simulation_end);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            simulation_time_ms += one_run;
        }
        simulation_time_ms /= timed_runs;
    }
    catch (...) {
        cudaEventDestroy(simulation_start);
        cudaEventDestroy(simulation_end);
        cudaFree(d_state);
        throw;
    }

    cudaEventDestroy(simulation_start);
    cudaEventDestroy(simulation_end);

    // ------------------------------------------------------------
    // Copy result back to CPU
    // ------------------------------------------------------------

    std::vector<Complex<T>> state(state_size);

    status = cudaMemcpy(state.data(), d_state, state_bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cudaFree(d_state);
        throw std::runtime_error(std::string("Final state copy failed: ") + cudaGetErrorString(status));
    }

    cudaFree(d_state);

    long double norm = 0.0L;
    for (const auto& amplitude : state) {
        norm += static_cast<long double>(amplitude.real) * amplitude.real
              + static_cast<long double>(amplitude.imag) * amplitude.imag;
    }

    std::cout << "Circuit load time   : " << load_time_ms << " ms\n"
              << "Qubits              : " << num_qubits << '\n'
              << "Gates               : " << num_gates << '\n'
              << "Precision           : "
              << (sizeof(T) == 4 ? "FP32/complex64" : "FP64/complex128") << '\n'
              << "GPU simulation time : " << simulation_time_ms << " ms\n";

    const long double norm_tolerance = sizeof(T) == 4 ? 1e-5L : 1e-12L;
    if (!std::isfinite(static_cast<double>(norm)) ||
        std::abs(norm - 1.0L) > norm_tolerance) {
        std::cout << "Final state norm    : " << std::setprecision(12)
                  << static_cast<double>(norm) << '\n';
    }

    if (!output_file.empty()) {
        const auto output_path = std::filesystem::path(output_file);
        if (output_path.has_parent_path())
            std::filesystem::create_directories(output_path.parent_path());

        if (output_path.extension() == ".bin") {
            // QUV1: 4 magic + uint32 dtype flag (0=fp32, 1=fp64) + 8 pad, then raw amps.
            char head[16] = {'Q', 'U', 'V', '1'};
            uint32_t flag = sizeof(T) == 4 ? 0 : 1;
            std::memcpy(head + 4, &flag, 4);
            std::ofstream output(output_file, std::ios::binary);
            if (!output)
                throw std::runtime_error("Cannot open output file: " + output_file);
            output.write(head, sizeof head);
            output.write(reinterpret_cast<const char*>(state.data()), state_bytes);
            output.flush();
            if (!output)
                throw std::runtime_error("Failed to write output file: " + output_file);
        } else {
            std::ofstream output(output_file);
            if (!output)
                throw std::runtime_error("Cannot open output file: " + output_file);

            output << std::setprecision(sizeof(T) == 4 ? 9 : 17)
                   << "# index real imag\n";
            for (uint64_t i = 0; i < state_size; ++i)
                output << i << ' ' << state[i].real << ' ' << state[i].imag << '\n';

            output.flush();
            if (!output)
                throw std::runtime_error("Failed to write output file: " + output_file);
        }
    }

    return 0;
}


int main(int argc, char** argv)
{
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <circuit.circuit> <fp32|fp64> [output.txt]\n";
            return 1;
        }

        const std::string circuit_file = argv[1];
        const std::string precision = argv[2];
        const std::string output_file = argc == 4 ? argv[3] : "";

        if (precision == "fp32")
            return run<float>(circuit_file, output_file);

        if (precision == "fp64")
            return run<double>(circuit_file, output_file);

        throw std::runtime_error("Precision must be fp32 or fp64");
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
