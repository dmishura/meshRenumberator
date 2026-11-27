#pragma once

#include "gs_m74d.H"
#include <cstdint>
#include <immintrin.h>

namespace gsExperiment {

// Variant: 0 = scalar(0,1)+packed(2,3), 1 = packed(0,1)+scalar(2,3),
// 2 = scalar(0)+packed(1,2)+scalar(3). The ordinary path is one inline-asm
// chunk so the selected mask offset and GPR allocation are explicit.
template<int Variant>
__attribute__((target("avx2,no-fma"), always_inline))
inline void cacheHybridRowsM8(double* x, double* bPrime, const int* starts,
    const int* indices, const double* coefficients, const double* diag,
    int first, int last)
{
    static_assert(Variant >= 0 && Variant <= 2);
    register const int64_t* maskBase asm("r15") = overlappingMasks;
    register const int32_t zero asm("r12") = 0;

    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell];
        const int end=starts[cell+1];
        double value=bPrime[cell];
        for (int face=begin; face<end; face+=4) {
            int64_t tmp;
            __m128d p0, p1, p2, p3, pair;
            const uint64_t faceIndex=static_cast<uint64_t>(face);

            if constexpr (Variant == 0) {
                asm volatile(
                    // Scalar products for slots 0 and 1.
                    "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p0]\n\t"
                    "vmulsd 0(%[co],%[face],8), %[p0], %[p0]\n\t"
                    "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p1]\n\t"
                    "vmulsd 8(%[co],%[face],8), %[p1], %[p1]\n\t"
                    // One packed product for slots 2 and 3.
                    "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[pair]\n\t"
                    "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovhpd (%[psi],%[tmp],8), %[pair], %[pair]\n\t"
                    "vmulpd 16(%[co],%[face],8), %[pair], %[pair]\n\t"
                    "leal 4(%k[face]), %k[tmp]\n\t"
                    "subl %k[end], %k[tmp]\n\t"
                    "cmovs %k[zero], %k[tmp]\n\t"
                    "vandpd 0(%[mask],%[tmp],8), %[p0], %[p0]\n\t"
                    "vandpd 8(%[mask],%[tmp],8), %[p1], %[p1]\n\t"
                    "vandpd 16(%[mask],%[tmp],8), %[pair], %[pair]\n\t"
                    "vsubsd %[p0], %[acc], %[acc]\n\t"
                    "vsubsd %[p1], %[acc], %[acc]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]\n\t"
                    "vunpckhpd %[pair], %[pair], %[pair]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]"
                    : [tmp] "=&r" (tmp), [p0] "=&x" (p0),
                      [p1] "=&x" (p1), [pair] "=&x" (pair),
                      [acc] "+x" (value)
                    : [ix] "r" (indices), [psi] "r" (x),
                      [co] "r" (coefficients), [mask] "r" (maskBase),
                      [face] "r" (faceIndex), [end] "r" (end),
                      [zero] "r" (zero)
                    : "cc", "memory");
            } else if constexpr (Variant == 1) {
                asm volatile(
                    // One packed product for slots 0 and 1.
                    "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[pair]\n\t"
                    "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovhpd (%[psi],%[tmp],8), %[pair], %[pair]\n\t"
                    "vmulpd 0(%[co],%[face],8), %[pair], %[pair]\n\t"
                    // Scalar products for slots 2 and 3.
                    "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p2]\n\t"
                    "vmulsd 16(%[co],%[face],8), %[p2], %[p2]\n\t"
                    "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p3]\n\t"
                    "vmulsd 24(%[co],%[face],8), %[p3], %[p3]\n\t"
                    "leal 4(%k[face]), %k[tmp]\n\t"
                    "subl %k[end], %k[tmp]\n\t"
                    "cmovs %k[zero], %k[tmp]\n\t"
                    "vandpd 0(%[mask],%[tmp],8), %[pair], %[pair]\n\t"
                    "vandpd 16(%[mask],%[tmp],8), %[p2], %[p2]\n\t"
                    "vandpd 24(%[mask],%[tmp],8), %[p3], %[p3]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]\n\t"
                    "vunpckhpd %[pair], %[pair], %[pair]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]\n\t"
                    "vsubsd %[p2], %[acc], %[acc]\n\t"
                    "vsubsd %[p3], %[acc], %[acc]"
                    : [tmp] "=&r" (tmp), [p2] "=&x" (p2),
                      [p3] "=&x" (p3), [pair] "=&x" (pair),
                      [acc] "+x" (value)
                    : [ix] "r" (indices), [psi] "r" (x),
                      [co] "r" (coefficients), [mask] "r" (maskBase),
                      [face] "r" (faceIndex), [end] "r" (end),
                      [zero] "r" (zero)
                    : "cc", "memory");
            } else {
                asm volatile(
                    // Scalar slot 0.
                    "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p0]\n\t"
                    "vmulsd 0(%[co],%[face],8), %[p0], %[p0]\n\t"
                    // One packed product for slots 1 and 2.
                    "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[pair]\n\t"
                    "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovhpd (%[psi],%[tmp],8), %[pair], %[pair]\n\t"
                    "vmulpd 8(%[co],%[face],8), %[pair], %[pair]\n\t"
                    // Scalar slot 3.
                    "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
                    "vmovsd (%[psi],%[tmp],8), %[p3]\n\t"
                    "vmulsd 24(%[co],%[face],8), %[p3], %[p3]\n\t"
                    "leal 4(%k[face]), %k[tmp]\n\t"
                    "subl %k[end], %k[tmp]\n\t"
                    "cmovs %k[zero], %k[tmp]\n\t"
                    "vandpd 0(%[mask],%[tmp],8), %[p0], %[p0]\n\t"
                    "vandpd 8(%[mask],%[tmp],8), %[pair], %[pair]\n\t"
                    "vandpd 24(%[mask],%[tmp],8), %[p3], %[p3]\n\t"
                    "vsubsd %[p0], %[acc], %[acc]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]\n\t"
                    "vunpckhpd %[pair], %[pair], %[pair]\n\t"
                    "vsubsd %[pair], %[acc], %[acc]\n\t"
                    "vsubsd %[p3], %[acc], %[acc]"
                    : [tmp] "=&r" (tmp), [p0] "=&x" (p0),
                      [p3] "=&x" (p3), [pair] "=&x" (pair),
                      [acc] "+x" (value)
                    : [ix] "r" (indices), [psi] "r" (x),
                      [co] "r" (coefficients), [mask] "r" (maskBase),
                      [face] "r" (faceIndex), [end] "r" (end),
                      [zero] "r" (zero)
                    : "cc", "memory");
            }
        }

        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

template<int Variant>
__attribute__((target("avx2,no-fma"), noinline))
inline void cacheHybridM8(Foam::scalarField& psi,
    const smootherTest::gamg::GAMGMatrix& matrix,
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
        cacheHybridRowsM8<Variant>(x,bPrime,starts,indices,coefficients,
                                   diag,0,data.terminalRow);
        scalarProductTerminalM74d(x,bPrime,starts,indices,coefficients,
                                  diag,data.terminalRow,n,faces);
    }
}

} // namespace gsExperiment
