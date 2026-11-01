#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define CSV_CHECK(x) do { auto s=(x); if(s!=CUSTATEVEC_STATUS_SUCCESS) throw std::runtime_error("cuStateVec status="+std::to_string((int)s)); } while(0)

template<class R> struct Traits;
template<> struct Traits<float>{ using C=cuFloatComplex; static constexpr cudaDataType_t dt=CUDA_C_32F; static constexpr custatevecComputeType_t ct=CUSTATEVEC_COMPUTE_32F; static C make(float r,float i){return make_cuFloatComplex(r,i);} };
template<> struct Traits<double>{ using C=cuDoubleComplex; static constexpr cudaDataType_t dt=CUDA_C_64F; static constexpr custatevecComputeType_t ct=CUSTATEVEC_COMPUTE_64F; static C make(double r,double i){return make_cuDoubleComplex(r,i);} };

enum class GateKind { U, CX };

struct Gate {
    GateKind kind;
    int32_t target = -1;
    int32_t control = -1;
    std::array<std::complex<double>,4> matrix{};
};

struct Circuit {
    int num_qubits = 0;
    std::complex<double> global_phase{1.0,0.0};
    std::vector<Gate> gates;
};

Circuit load_circuit(const std::string& path){
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open circuit file: "+path);
    std::string word; Circuit c; size_t gate_count=0;
    if(!(f>>word)||word!="QISKIT_CIRCUIT_V1") throw std::runtime_error("invalid circuit IR header");
    if(!(f>>word>>c.num_qubits)||word!="qubits"||c.num_qubits<=0||c.num_qubits>=63)
        throw std::runtime_error("invalid qubit count in circuit IR");
    double phase_real=0,phase_imag=0;
    if(!(f>>word>>phase_real>>phase_imag)||word!="global_phase")
        throw std::runtime_error("invalid global phase in circuit IR");
    c.global_phase={phase_real,phase_imag};
    if(!(f>>word>>gate_count)||word!="gates") throw std::runtime_error("invalid gate count in circuit IR");
    c.gates.reserve(gate_count);
    for(size_t i=0;i<gate_count;i++){
        if(!(f>>word)) throw std::runtime_error("unexpected end of circuit IR");
        Gate g;
        if(word=="u"){
            g.kind=GateKind::U;
            if(!(f>>g.target)) throw std::runtime_error("invalid u gate in circuit IR");
            for(auto& z:g.matrix){ double real=0,imag=0; if(!(f>>real>>imag)) throw std::runtime_error("invalid u matrix in circuit IR"); z={real,imag}; }
        } else if(word=="cx"){
            g.kind=GateKind::CX;
            if(!(f>>g.control>>g.target)) throw std::runtime_error("invalid cx gate in circuit IR");
        } else throw std::runtime_error("unsupported circuit IR operation: "+word);
        if(g.target<0||g.target>=c.num_qubits||
           (g.kind==GateKind::CX&&(g.control<0||g.control>=c.num_qubits||g.control==g.target)))
            throw std::runtime_error("invalid qubit index in circuit IR");
        c.gates.push_back(g);
    }
    if(f>>word) throw std::runtime_error("unexpected trailing data in circuit IR");
    return c;
}

template<class R>
std::array<typename Traits<R>::C,4> cv(const std::array<std::complex<double>,4>& a){
    std::array<typename Traits<R>::C,4> b{};
    for(int i=0;i<4;i++) b[i]=Traits<R>::make(a[i].real(),a[i].imag());
    return b;
}

template<class R>
void simulate(const Circuit& c,const std::string& out){
    using T=Traits<R>; using C=typename T::C;
    if(c.num_qubits>=63) throw std::runtime_error("starter requires <63 qubits");
    uint64_t n=uint64_t(1)<<c.num_qubits; size_t bytes=n*sizeof(C);
    std::vector<C> h(n,T::make(0,0)); h[0]=T::make(R(c.global_phase.real()),R(c.global_phase.imag()));
    C* d=nullptr; CUDA_CHECK(cudaMalloc(&d,bytes)); CUDA_CHECK(cudaMemcpy(d,h.data(),bytes,cudaMemcpyHostToDevice));
    custatevecHandle_t handle; CSV_CHECK(custatevecCreate(&handle));
    void* ws=nullptr; size_t ws_cap=0;

    auto apply=[&](const std::array<C,4>& m,int target,const int32_t* ctrl,uint32_t nc){
        int32_t targets[]={target}; size_t need=0;
        CSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(handle,T::dt,c.num_qubits,m.data(),T::dt,
            CUSTATEVEC_MATRIX_LAYOUT_ROW,0,1,nc,T::ct,&need));
        if(need>ws_cap){ if(ws) CUDA_CHECK(cudaFree(ws)); CUDA_CHECK(cudaMalloc(&ws,need)); ws_cap=need; }
        CSV_CHECK(custatevecApplyMatrix(handle,d,T::dt,c.num_qubits,m.data(),T::dt,
            CUSTATEVEC_MATRIX_LAYOUT_ROW,0,targets,1,ctrl,nullptr,nc,T::ct,ws,need));
    };

    const auto X=cv<R>({0,1,1,0});
    auto reset=[&](){ CUDA_CHECK(cudaMemcpy(d,h.data(),bytes,cudaMemcpyHostToDevice)); };
    auto run_circuit=[&](){
        for(const auto& g:c.gates){
            if(g.kind==GateKind::CX){ int32_t ctrl[]={g.control}; apply(X,g.target,ctrl,1); }
            else apply(cv<R>(g.matrix),g.target,nullptr,0);
        }
    };

    // 1 warm-up (workspace allocation + JIT) + 3 timed runs, mean.
    reset(); run_circuit();
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    float ms=0;
    for(int rep=0;rep<3;++rep){
        reset(); CUDA_CHECK(cudaEventRecord(a)); run_circuit();
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float one=0; CUDA_CHECK(cudaEventElapsedTime(&one,a,b)); ms+=one;
    }
    ms/=3;
    CUDA_CHECK(cudaMemcpy(h.data(),d,bytes,cudaMemcpyDeviceToHost));
    long double norm=0; for(auto z:h) norm+=(long double)z.x*z.x+(long double)z.y*z.y;
    std::cout<<"Qubits              : "<<c.num_qubits<<"\nGates               : "<<c.gates.size()
             <<"\nPrecision           : "<<(sizeof(R)==4?"FP32/complex64":"FP64/complex128")
             <<"\nGPU simulation time : "<<ms<<" ms\n";
    const long double norm_tolerance=sizeof(R)==4?1e-5L:1e-12L;
    if(!std::isfinite((double)norm)||std::abs(norm-1.0L)>norm_tolerance)
        std::cout<<"Final state norm    : "<<std::setprecision(12)<<(double)norm<<"\n";
    if(!out.empty()){
        auto p=std::filesystem::path(out); if(p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        if(p.extension()==".bin"){
            // QUV1: 4 magic + uint32 dtype flag (0=fp32, 1=fp64) + 8 pad, then raw amps.
            char head[16]={'Q','U','V','1'}; uint32_t flag=sizeof(R)==4?0:1; std::memcpy(head+4,&flag,4);
            std::ofstream f(out,std::ios::binary);
            if(!f) throw std::runtime_error("cannot open output file: "+out);
            f.write(head,sizeof head); f.write(reinterpret_cast<const char*>(h.data()),bytes);
            f.flush(); if(!f) throw std::runtime_error("failed to write output file: "+out);
        } else {
            std::ofstream f(out);
            if(!f) throw std::runtime_error("cannot open output file: "+out);
            f<<std::setprecision(sizeof(R)==4?9:17)<<"# index real imag\n";
            for(uint64_t i=0;i<n;i++) f<<i<<" "<<h[i].x<<" "<<h[i].y<<"\n";
            f.flush();
            if(!f) throw std::runtime_error("failed to write output file: "+out);
        }
    }
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b)); if(ws) CUDA_CHECK(cudaFree(ws));
    CSV_CHECK(custatevecDestroy(handle)); CUDA_CHECK(cudaFree(d));
}

int main(int argc,char** argv){
    try{
        if(argc<3||argc>4){ std::cerr<<"Usage: "<<argv[0]<<" <circuit.circuit> <fp32|fp64> [output.txt]\n"; return 1; }
        auto t0=std::chrono::high_resolution_clock::now(); Circuit c=load_circuit(argv[1]); auto t1=std::chrono::high_resolution_clock::now();
        std::cout<<"Circuit load time   : "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<" ms\n";
        std::string p=argv[2], out=argc==4?argv[3]:"";
        if(p=="fp32") simulate<float>(c,out); else if(p=="fp64") simulate<double>(c,out); else throw std::runtime_error("precision must be fp32 or fp64");
    } catch(const std::exception& e){ std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
}
