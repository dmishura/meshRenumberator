#pragma once
#include "gs_m7.H"
#include <cstdint>
#include <immintrin.h>

namespace gsExperiment {

// Preserve M7's existing masked-load Terminal algorithm while keeping its
// registers out of the ordinary-row allocator.
__attribute__((target("avx2,no-fma"), noinline))
inline void scalarProductTerminalM74d(double* x, double* bPrime,
    const int* starts, const int* indices, const double* coefficients,
    const double* diag, int first, int last, int faces)
{
    scalarProductRows<true>(x,bPrime,starts,indices,coefficients,diag,
                           first,last,faces);
}

// M7.4d keeps M7's four independent scalar products, but uses M6.4d's
// overlapping cache-line mask stream and direct SIB addressing. The main
// body loads only full four-face chunks; the suffix is delegated to the
// unchanged M6.4d terminal helper.
__attribute__((target("avx2,no-fma"), always_inline))
inline void scalarProductRowsM74d(double* x, double* bPrime,
    const int* starts, const int* indices, const double* coefficients,
    const double* diag, int first, int last)
{
    register const int64_t* maskBase asm("r15") = overlappingMasks;
    register const int32_t zero asm("r12") = 0;

    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell];
        const int end=starts[cell+1];
        double value=bPrime[cell];

        for (int face=begin; face<end; face+=4) {
            int64_t indexOrMaskOffset;
            __m128d p0, p1, p2, p3;
            const uint64_t faceIndex=static_cast<uint64_t>(face);
            asm volatile(
                // Reuse one GPR for the four signed 32-bit neighbour indices.
                "movslq 0(%[indices],%[face],4), %[tmp]\n\t"
                "vmovsd (%[psi],%[tmp],8), %[p0]\n\t"
                "vmulsd 0(%[coeff],%[face],8), %[p0], %[p0]\n\t"
                "movslq 4(%[indices],%[face],4), %[tmp]\n\t"
                "vmovsd (%[psi],%[tmp],8), %[p1]\n\t"
                "vmulsd 8(%[coeff],%[face],8), %[p1], %[p1]\n\t"
                "movslq 8(%[indices],%[face],4), %[tmp]\n\t"
                "vmovsd (%[psi],%[tmp],8), %[p2]\n\t"
                "vmulsd 16(%[coeff],%[face],8), %[p2], %[p2]\n\t"
                "movslq 12(%[indices],%[face],4), %[tmp]\n\t"
                "vmovsd (%[psi],%[tmp],8), %[p3]\n\t"
                "vmulsd 24(%[coeff],%[face],8), %[p3], %[p3]\n\t"
                // offset=max(0, face+4-end); the zero is held in r12d.
                "leal 4(%k[face]), %k[tmp]\n\t"
                "subl %k[end], %k[tmp]\n\t"
                "cmovs %k[zero], %k[tmp]\n\t"
                // Four scalar masks are read from the same 64-byte table.
                "vandpd 0(%[mask],%[tmp],8), %[p0], %[p0]\n\t"
                "vandpd 8(%[mask],%[tmp],8), %[p1], %[p1]\n\t"
                "vandpd 16(%[mask],%[tmp],8), %[p2], %[p2]\n\t"
                "vandpd 24(%[mask],%[tmp],8), %[p3], %[p3]\n\t"
                // Preserve Original GS's sequential subtraction order.
                "vsubsd %[p0], %[acc], %[acc]\n\t"
                "vsubsd %[p1], %[acc], %[acc]\n\t"
                "vsubsd %[p2], %[acc], %[acc]\n\t"
                "vsubsd %[p3], %[acc], %[acc]"
                : [tmp] "=&r" (indexOrMaskOffset),
                  [p0] "=&x" (p0), [p1] "=&x" (p1),
                  [p2] "=&x" (p2), [p3] "=&x" (p3),
                  [acc] "+x" (value)
                : [indices] "r" (indices), [psi] "r" (x),
                  [coeff] "r" (coefficients), [mask] "r" (maskBase),
                  [face] "r" (faceIndex), [end] "r" (end),
                  [zero] "r" (zero)
                : "cc", "memory");
        }

        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void scalarProductMinimal4M74d(Foam::scalarField& psi,
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
        scalarProductRowsM74d(x,bPrime,starts,indices,coefficients,diag,
                              0,data.terminalRow);
        scalarProductTerminalM74d(x,bPrime,starts,indices,coefficients,
                                  diag,data.terminalRow,n,faces);
    }
}

} // namespace gsExperiment
