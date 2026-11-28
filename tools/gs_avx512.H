#pragma once
#include "gs_minimal_opt.H"

namespace gsExperiment {
alignas(8) inline constexpr unsigned char tailMasks4[]={15,7,3,1,0};

template<bool Compact>
__attribute__((target("avx512vl,avx512dq,no-fma"), always_inline))
inline double avx512Chunk(double value,long face,int end,const int* indices,
                          const double* psi,const double* coefficients)
{
    long tmp;
    if constexpr (!Compact) {
        asm(
            "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[x],%[tmp],8), %%xmm16\n\t"
            "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[x],%[tmp],8), %%xmm16, %%xmm16\n\t"
            "vmulpd 0(%[co],%[face],8), %%xmm16, %%xmm16\n\t"
            "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[x],%[tmp],8), %%xmm17\n\t"
            "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[x],%[tmp],8), %%xmm17, %%xmm17\n\t"
            "vmulpd 16(%[co],%[face],8), %%xmm17, %%xmm17\n\t"
            "leal 4(%k[face]), %k[tmp]\n\t"
            "subl %k[end], %k[tmp]\n\t"
            "cmovs %k[zero], %k[tmp]\n\t"
            "vandpd (%[mask],%[tmp],8), %%xmm16, %%xmm16\n\t"
            "vandpd 16(%[mask],%[tmp],8), %%xmm17, %%xmm17\n\t"
            "vsubsd %%xmm16, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm16, %%xmm16, %%xmm16\n\t"
            "vsubsd %%xmm16, %[acc], %[acc]\n\t"
            "vsubsd %%xmm17, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm17, %%xmm17, %%xmm17\n\t"
            "vsubsd %%xmm17, %[acc], %[acc]\n\t"
            : [acc] "+x"(value), [tmp] "=&r"(tmp)
            : [ix] "r"(indices), [face] "r"(face), [x] "r"(psi),
              [co] "r"(coefficients), [end] "r"(end), [zero] "r"(0),
              [mask] "r"(overlappingMasks)
            : "cc","memory","xmm16","xmm17");
    } else {
        asm(
            "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[x],%[tmp],8), %%xmm4\n\t"
            "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[x],%[tmp],8), %%xmm4, %%xmm4\n\t"
            "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[x],%[tmp],8), %%xmm5\n\t"
            "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[x],%[tmp],8), %%xmm5, %%xmm5\n\t"
            "leal 4(%k[face]), %k[tmp]\n\t"
            "subl %k[end], %k[tmp]\n\t"
            "cmovs %k[zero], %k[tmp]\n\t"
            "kmovb (%[mask],%[tmp]), %%k1\n\t"
            "vinsertf128 $1, %%xmm5, %%ymm4, %%ymm4\n\t"
            "vmulpd (%[co],%[face],8), %%ymm4, %%ymm4%{%%k1%}%{z%}\n\t"
            "vextractf128 $1, %%ymm4, %%xmm5\n\t"
            "vsubsd %%xmm4, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm4, %%xmm4, %%xmm4\n\t"
            "vsubsd %%xmm4, %[acc], %[acc]\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm5, %%xmm5, %%xmm5\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            : [acc] "+x"(value), [tmp] "=&r"(tmp)
            : [ix] "r"(indices), [face] "r"(face), [x] "r"(psi),
              [co] "r"(coefficients), [end] "r"(end), [zero] "r"(0),
              [mask] "r"(tailMasks4)
            : "cc","memory","xmm4","xmm5","k1");
    }
    return value;
}

template<bool Compact>
__attribute__((target("avx512vl,avx512dq,no-fma"), optimize("fp-contract=off"), noinline))
inline void minimal4Avx512(
    Foam::scalarField& psi, const smootherTest::gamg::GAMGMatrix& matrix,
    const Foam::scalarField& source, int sweeps, MinimalOptData& data)
{
    const int n=matrix.nCells(), faces=matrix.nFaces();
    const int* starts=matrix.ownerStartAddr().data();
    const int* indices=matrix.upperAddr().data();
    const double* coefficients=matrix.upper().data();
    const double* diag=matrix.diag().data();
    const double* rhs=source.data();
    double* x=psi.data();
    double* bPrime=data.bPrime.data();
    for (int sweep=0; sweep<sweeps; ++sweep) {
        std::copy(rhs,rhs+n,bPrime);
        const int terminalRow=data.terminalRow;
        for (int cell=0; cell<terminalRow; ++cell) {
            const int begin=starts[cell], end=starts[cell+1];
            double value=bPrime[cell];
            for (int face=begin; face<end; face+=4)
                value=avx512Chunk<Compact>(value,face,end,indices,x,coefficients);
            value/=diag[cell];
            for (int face=begin; face<end; ++face)
                bPrime[indices[face]]-=coefficients[face]*value;
            x[cell]=value;
        }
        optimizedTerminalRowsIsolated(x,bPrime,starts,indices,coefficients,
            diag,data.terminalRow,n,faces);
    }
}

} // namespace gsExperiment
