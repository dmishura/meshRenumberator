#pragma once
#include "gs_minimal4.H"
#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace gsExperiment {
struct MinimalOptData {
    Foam::scalarField bPrime;
    int terminalRow;

    explicit MinimalOptData(const smootherTest::gamg::GAMGMatrix& matrix)
      : bPrime(matrix.nCells()), terminalRow(0)
    {
        const auto& starts=matrix.ownerStartAddr();
        if (matrix.nFaces()>=4)
            terminalRow=std::upper_bound(starts.begin()+1,starts.end(),matrix.nFaces()-3)
                       -starts.begin()-1;
    }
};

// M6.1/M6.2 experiment: one overlapping mask stream, indexed in 8-byte
// units. 64-byte alignment ensures both unaligned 16-byte loads for every
// legal offset 0–3 stay within one cache line. The maximum second-half load
// starts at entry 5 and reads entries 5–6, within the initialized table.
alignas(64) static constexpr int64_t overlappingMasks[8] = {
    -1, -1, -1, -1, 0, 0, 0, 0
};

// Ordinary loads are legal for rows ending at or before nFaces-3.
// A separate terminal range uses masked loads from the same original arrays.
// This preserves matrix sizes and needs no copied LDU storage or tail allocation.
template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRows(double* x, double* bPrime, const int* starts,
                         const int* indices, const double* coefficients,
                         const double* diag, int first, int last, int faces)
{
    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell], end=starts[cell+1];
        double value=bPrime[cell];
        for (int face=begin; face<end; face+=4) {
            const int count=std::min(4,end-face);
            const int* ix=indices+face;
            const double* co=coefficients+face;
            __m256d loaded;
            __m128i loadedIndices;
            if constexpr (Terminal) {
                const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                                  _mm_setr_epi32(0,1,2,3));
                loadedIndices=_mm_maskload_epi32(ix,safe);
                loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
            }
            const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
            const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
            const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
            const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
            const double* mask=reinterpret_cast<const double*>(activeMasks[count]);
            const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
            const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
            const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
            const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
            const __m128d p01=_mm_and_pd(_mm_mul_pd(x01,co01),_mm_loadu_pd(mask));
            const __m128d p23=_mm_and_pd(_mm_mul_pd(x23,co23),_mm_loadu_pd(mask+2));
            value-=_mm_cvtsd_f64(p01);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
            value-=_mm_cvtsd_f64(p23);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
        }
        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4(Foam::scalarField& psi,
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
        optimizedRows<false>(x,bPrime,starts,indices,coefficients,diag,0,data.terminalRow,faces);
        optimizedRows<true>(x,bPrime,starts,indices,coefficients,diag,data.terminalRow,n,faces);
    }
}

// M6.1 keeps M6's loop, loads, arithmetic and scatter unchanged. Only mask
// selection changes from activeMasks[count] to the overlapping 8-byte stream.
template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRowsOverlappingMasks(double* x, double* bPrime,
                         const int* starts, const int* indices,
                         const double* coefficients, const double* diag,
                         int first, int last, int faces)
{
    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell], end=starts[cell+1];
        double value=bPrime[cell];
        for (int face=begin; face<end; face+=4) {
            const int count=std::min(4,end-face);
            const int64_t* mask=overlappingMasks+(4-count);
            const int* ix=indices+face;
            const double* co=coefficients+face;
            __m256d loaded;
            __m128i loadedIndices;
            if constexpr (Terminal) {
                const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                                  _mm_setr_epi32(0,1,2,3));
                loadedIndices=_mm_maskload_epi32(ix,safe);
                loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
            }
            const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
            const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
            const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
            const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
            const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
            const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
            const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
            const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
            const __m128d p01=_mm_and_pd(_mm_mul_pd(x01,co01),
                _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask))));
            const __m128d p23=_mm_and_pd(_mm_mul_pd(x23,co23),
                _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask+2))));
            value-=_mm_cvtsd_f64(p01);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
            value-=_mm_cvtsd_f64(p23);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
        }
        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4OverlappingMasks(Foam::scalarField& psi,
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
        optimizedRowsOverlappingMasks<false>(x,bPrime,starts,indices,coefficients,
            diag,0,data.terminalRow,faces);
        optimizedRowsOverlappingMasks<true>(x,bPrime,starts,indices,coefficients,
            diag,data.terminalRow,n,faces);
    }
}

// M6.2: use the clamped chunk-end distance directly as the overlapping-mask
// table index; no `count = min(4, remaining)` temporary is formed.
template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRowsSimplifiedMaskIndex(double* __restrict__ x,
                         double* __restrict__ bPrime,
                         const int* __restrict__ starts,
                         const int* __restrict__ indices,
                         const double* __restrict__ coefficients,
                         const double* __restrict__ diag,
                         int first, int last, int faces)
{
    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell], end=starts[cell+1];
        double value=bPrime[cell];
        for (int face=begin; face<end; face+=4) {
            const int* ix=indices+face;
            const double* co=coefficients+face;
            __m256d loaded;
            __m128i loadedIndices;
            if constexpr (Terminal) {
                const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                                  _mm_setr_epi32(0,1,2,3));
                loadedIndices=_mm_maskload_epi32(ix,safe);
                loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
            }
            const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
            const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
            const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
            const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
            const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
            const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
            const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
            const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
            __m128d p01=_mm_mul_pd(x01,co01);
            __m128d p23=_mm_mul_pd(x23,co23);
            // Form max(0, face + 4 - end) and consume it directly as the SIB
            // index in both mask loads. No count or temporary mask pointer.
            const std::uint32_t zero=0;
            std::uint64_t maskOffset;
            asm volatile(
                "leal 4(%k[face]), %k[offset]\n\t"
                "subl %k[end], %k[offset]\n\t"
                "cmovs %k[zero], %k[offset]\n\t"
                "vandpd (%[base],%[offset],8), %[lo], %[lo]\n\t"
                "vandpd 16(%[base],%[offset],8), %[hi], %[hi]"
                : [offset] "=&r" (maskOffset), [lo] "+x" (p01),
                  [hi] "+x" (p23)
                : [base] "r" (overlappingMasks), [face] "r" (face),
                  [end] "r" (end), [zero] "r" (zero)
                : "cc", "memory");
            value-=_mm_cvtsd_f64(p01);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
            value-=_mm_cvtsd_f64(p23);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
        }
        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4SimplifiedMaskIndex(Foam::scalarField& psi,
    const smootherTest::gamg::GAMGMatrix& matrix,
    const Foam::scalarField& source, int sweeps, MinimalOptData& data)
{
    const int n=matrix.nCells(), faces=matrix.nFaces();
    const int* __restrict__ starts=matrix.ownerStartAddr().data();
    const int* __restrict__ indices=matrix.upperAddr().data();
    const double* __restrict__ coefficients=matrix.upper().data();
    const double* __restrict__ diag=matrix.diag().data();
    const double* __restrict__ rhs=source.data();
    double* __restrict__ x=psi.data();
    double* __restrict__ bPrime=data.bPrime.data();
    for (int sweep=0; sweep<sweeps; ++sweep) {
        std::copy(rhs,rhs+n,bPrime);
        optimizedRowsSimplifiedMaskIndex<false>(x,bPrime,starts,indices,coefficients,
            diag,0,data.terminalRow,faces);
        optimizedRowsSimplifiedMaskIndex<true>(x,bPrime,starts,indices,coefficients,
            diag,data.terminalRow,n,faces);
    }
}

// Load four indexed psi values while reusing one architectural GPR for all
// four sign-extended indices. The repeated writes to tmp are independent
// renamed versions; the memory dependencies remain each index -> psi load.
__attribute__((target("avx2,no-fma"), always_inline))
inline void loadPsiPairsSingleIndexGpr(
    const int* indices, std::uint64_t face, const double* psi,
    __m128d& x01, __m128d& x23)
{
    std::int64_t tmp;
    asm volatile(
        "movslq 0(%[indices],%[face],4), %[tmp]\n\t"
        "vmovsd (%[psi],%[tmp],8), %[x01]\n\t"
        "movslq 4(%[indices],%[face],4), %[tmp]\n\t"
        "vmovhpd (%[psi],%[tmp],8), %[x01], %[x01]\n\t"
        "movslq 8(%[indices],%[face],4), %[tmp]\n\t"
        "vmovsd (%[psi],%[tmp],8), %[x23]\n\t"
        "movslq 12(%[indices],%[face],4), %[tmp]\n\t"
        "vmovhpd (%[psi],%[tmp],8), %[x23], %[x23]"
        : [tmp] "=&r" (tmp), [x01] "=&x" (x01), [x23] "=&x" (x23)
        : [indices] "r" (indices), [face] "r" (face), [psi] "r" (psi)
        : "memory");
}

// M6.4 row implementation. Terminal handling deliberately remains delegated
// to the unchanged M6/M6.2 paths in the public wrappers below.
template<bool OverlappingMasks, bool MemoryZero = false>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRowsSingleIndexGpr(
    double* x, double* bPrime, const int* starts, const int* indices,
    const double* coefficients, const double* diag, int first, int last)
{
    if constexpr (OverlappingMasks) {
        register const int64_t* maskBase asm("r15") = overlappingMasks;
        for (int cell=first; cell<last; ++cell) {
            const int begin=starts[cell], end=starts[cell+1];
            double value=bPrime[cell];
            for (int face=begin; face<end; face+=4) {
                const double* co=coefficients+face;
                __m128d x01, x23;
                loadPsiPairsSingleIndexGpr(
                    indices,static_cast<std::uint64_t>(face),x,x01,x23);
                const __m128d co01=_mm_loadu_pd(co);
                const __m128d co23=_mm_loadu_pd(co+2);
                __m128d p01=_mm_mul_pd(x01,co01);
                __m128d p23=_mm_mul_pd(x23,co23);
                std::uint64_t maskOffset;
                if constexpr (MemoryZero) {
                    asm volatile(
                        "leal 4(%k[face]), %k[offset]\n\t"
                        "subl %k[end], %k[offset]\n\t"
                        "cmovsl 32(%[base]), %k[offset]\n\t"
                        "vandpd (%[base],%[offset],8), %[lo], %[lo]\n\t"
                        "vandpd 16(%[base],%[offset],8), %[hi], %[hi]"
                        : [offset] "=&r" (maskOffset), [lo] "+x" (p01),
                          [hi] "+x" (p23)
                        : [base] "r" (maskBase), [face] "r" (face),
                          [end] "r" (end)
                        : "cc", "memory");
                } else {
                    const std::uint32_t zero=0;
                    asm volatile(
                        "leal 4(%k[face]), %k[offset]\n\t"
                        "subl %k[end], %k[offset]\n\t"
                        "cmovs %k[zero], %k[offset]\n\t"
                        "vandpd (%[base],%[offset],8), %[lo], %[lo]\n\t"
                        "vandpd 16(%[base],%[offset],8), %[hi], %[hi]"
                        : [offset] "=&r" (maskOffset), [lo] "+x" (p01),
                          [hi] "+x" (p23)
                        : [base] "r" (maskBase), [face] "r" (face),
                          [end] "r" (end), [zero] "r" (zero)
                        : "cc", "memory");
                }
                value-=_mm_cvtsd_f64(p01);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
                value-=_mm_cvtsd_f64(p23);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
            }
            value/=diag[cell];
            for (int face=begin; face<end; ++face)
                bPrime[indices[face]]-=coefficients[face]*value;
            x[cell]=value;
        }
    } else {
        for (int cell=first; cell<last; ++cell) {
            const int begin=starts[cell], end=starts[cell+1];
            double value=bPrime[cell];
            for (int face=begin; face<end; face+=4) {
                const int count=std::min(4,end-face);
                const double* co=coefficients+face;
                __m128d x01, x23;
                loadPsiPairsSingleIndexGpr(
                    indices,static_cast<std::uint64_t>(face),x,x01,x23);
                const double* mask=reinterpret_cast<const double*>(activeMasks[count]);
                const __m128d p01=_mm_and_pd(
                    _mm_mul_pd(x01,_mm_loadu_pd(co)),_mm_loadu_pd(mask));
                const __m128d p23=_mm_and_pd(
                    _mm_mul_pd(x23,_mm_loadu_pd(co+2)),_mm_loadu_pd(mask+2));
                value-=_mm_cvtsd_f64(p01);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
                value-=_mm_cvtsd_f64(p23);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
            }
            value/=diag[cell];
            for (int face=begin; face<end; ++face)
                bPrime[indices[face]]-=coefficients[face]*value;
            x[cell]=value;
        }
    }
}

template<bool OverlappingMasks>
__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4SingleIndexGpr(Foam::scalarField& psi,
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
        optimizedRowsSingleIndexGpr<OverlappingMasks, false>(
            x,bPrime,starts,indices,coefficients,diag,0,data.terminalRow);
        if constexpr (OverlappingMasks) {
            // Terminal rows retain the unchanged M6.2 safe-load path.
            optimizedRowsSimplifiedMaskIndex<true>(
                x,bPrime,starts,indices,coefficients,diag,
                data.terminalRow,n,faces);
        } else {
            // Terminal rows retain the unchanged M6 safe-load path.
            optimizedRows<true>(
                x,bPrime,starts,indices,coefficients,diag,
                data.terminalRow,n,faces);
        }
    }
}

// M6.4c uses the zero word in overlappingMasks as the CMOVS source. This
// keeps the mask base in r15 and avoids dedicating a GPR to the zero value.
__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4SingleIndexGprMemoryZero(Foam::scalarField& psi,
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
        optimizedRowsSingleIndexGpr<true, true>(
            x,bPrime,starts,indices,coefficients,diag,0,data.terminalRow);
        // Keep the existing M6.2 terminal implementation unchanged.
        optimizedRowsSimplifiedMaskIndex<true>(
            x,bPrime,starts,indices,coefficients,diag,
            data.terminalRow,n,faces);
    }
}

// Keep the existing Terminal implementation out of the main M6.4 register
// allocation. Its masked-load algorithm and row range are unchanged.
__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedTerminalRowsIsolated(double* x, double* bPrime,
    const int* starts, const int* indices, const double* coefficients,
    const double* diag, int first, int last, int faces)
{
    optimizedRowsSimplifiedMaskIndex<true>(x,bPrime,starts,indices,
        coefficients,diag,first,last,faces);
}

// M6.4d/e: M6.4b/c main loop with Terminal emitted as a separate call.
template<bool MemoryZero>
__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4SingleIndexGprIsolatedTerminal(
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
        optimizedRowsSingleIndexGpr<true, MemoryZero>(
            x,bPrime,starts,indices,coefficients,diag,0,data.terminalRow);
        optimizedTerminalRowsIsolated(x,bPrime,starts,indices,coefficients,
            diag,data.terminalRow,n,faces);
    }
}

// M6.2 alternative induction: track remaining faces instead of deriving the
// tail position from face/end. This is retained as a codegen candidate only.
template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRowsRemainingCount(double* x, double* bPrime,
                         const int* starts, const int* indices,
                         const double* coefficients, const double* diag,
                         int first, int last, int faces)
{
    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell], end=starts[cell+1];
        double value=bPrime[cell];
        int remaining=end-begin;
        for (int face=begin; remaining>0; face+=4, remaining-=4) {
            const int offset=std::max(0,4-remaining);
            const int64_t* mask=overlappingMasks+offset;
            const int* ix=indices+face;
            const double* co=coefficients+face;
            __m256d loaded;
            __m128i loadedIndices;
            if constexpr (Terminal) {
                const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                                  _mm_setr_epi32(0,1,2,3));
                loadedIndices=_mm_maskload_epi32(ix,safe);
                loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
            }
            const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
            const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
            const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
            const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
            const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
            const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
            const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
            const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
            const __m128d p01=_mm_and_pd(_mm_mul_pd(x01,co01),
                _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask))));
            const __m128d p23=_mm_and_pd(_mm_mul_pd(x23,co23),
                _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask+2))));
            value-=_mm_cvtsd_f64(p01);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
            value-=_mm_cvtsd_f64(p23);
            value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
        }
        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4RemainingCount(Foam::scalarField& psi,
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
        optimizedRowsRemainingCount<false>(x,bPrime,starts,indices,coefficients,
            diag,0,data.terminalRow,faces);
        optimizedRowsRemainingCount<true>(x,bPrime,starts,indices,coefficients,
            diag,data.terminalRow,n,faces);
    }
}

// Fixed-size lane masks for VMASKMOVPD.  Each low 64-bit element is copied
// from activeMasks; the high element is zero so every store touches one scalar.
struct MaskedScatterData {
    MinimalOptData base;
    alignas(16) __m128i laneMasks[5][4];

    explicit MaskedScatterData(const smootherTest::gamg::GAMGMatrix& matrix)
      : base(matrix)
    {
        for (int count=0; count<=4; ++count)
            for (int lane=0; lane<4; ++lane)
                laneMasks[count][lane] = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(activeMasks[count]+lane));
    }
};

__attribute__((target("avx2,no-fma"), always_inline))
inline void maskedScatter(double* dst, __m128d product, __m128i mask)
{
    const __m128d old = _mm_load_sd(dst);
    const __m128d updated = _mm_sub_sd(old, product);
    _mm_maskstore_pd(dst, mask, updated);
}

template<bool Terminal, bool Shared>
__attribute__((target("avx2,no-fma"), always_inline))
inline void maskedScatterChunk(double* bPrime, const int* indices,
                               const double* coefficients, int face, int count,
                               int faces, double value,
                               const __m128i (*laneMasks)[4])
{
    const int* ix = indices + face;
    const double* co = coefficients + face;
    __m128i packedIndices;
    __m256d packedCoefficients;
    if constexpr (Terminal) {
        const __m128i safe = _mm_cmpgt_epi32(
            _mm_set1_epi32(faces-face), _mm_setr_epi32(0,1,2,3));
        packedIndices = _mm_maskload_epi32(ix, safe);
        packedCoefficients = _mm256_maskload_pd(
            co, _mm256_cvtepi32_epi64(safe));
    } else {
        packedIndices = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ix));
        packedCoefficients = _mm256_loadu_pd(co);
    }

    const int i0 = _mm_cvtsi128_si32(packedIndices);
    const int i1 = _mm_extract_epi32(packedIndices, 1);
    const int i2 = _mm_extract_epi32(packedIndices, 2);
    const int i3 = _mm_extract_epi32(packedIndices, 3);
    const __m128d c01 = _mm256_castpd256_pd128(packedCoefficients);
    const __m128d c23 = _mm256_extractf128_pd(packedCoefficients, 1);
    const __m128i* masks = Shared ? laneMasks[count] : nullptr;
    const auto laneMask = [&](int lane) {
        if constexpr (Shared) return masks[lane];
        else return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(activeMasks[count]+lane));
    };

    if constexpr (Shared) {
        // Two independent vector products; stores remain in face order below.
        const __m128d valuePair = _mm_set1_pd(value);
        const __m128d p01 = _mm_mul_pd(c01, valuePair);
        const __m128d p23 = _mm_mul_pd(c23, valuePair);
        maskedScatter(bPrime+i0, p01, laneMask(0));
        maskedScatter(bPrime+i1, _mm_unpackhi_pd(p01,p01), laneMask(1));
        maskedScatter(bPrime+i2, p23, laneMask(2));
        maskedScatter(bPrime+i3, _mm_unpackhi_pd(p23,p23), laneMask(3));
    } else {
        // Basic is deliberately scalar so it provides a clean masked-store baseline.
        const double c0 = _mm_cvtsd_f64(c01);
        const double c1 = _mm_cvtsd_f64(_mm_unpackhi_pd(c01,c01));
        const double c2 = _mm_cvtsd_f64(c23);
        const double c3 = _mm_cvtsd_f64(_mm_unpackhi_pd(c23,c23));
        maskedScatter(bPrime+i0, _mm_mul_sd(_mm_set_sd(c0),_mm_set_sd(value)), laneMask(0));
        maskedScatter(bPrime+i1, _mm_mul_sd(_mm_set_sd(c1),_mm_set_sd(value)), laneMask(1));
        maskedScatter(bPrime+i2, _mm_mul_sd(_mm_set_sd(c2),_mm_set_sd(value)), laneMask(2));
        maskedScatter(bPrime+i3, _mm_mul_sd(_mm_set_sd(c3),_mm_set_sd(value)), laneMask(3));
    }
}

template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void maskedGatherChunk(double& value, const double* x,
                              const int* indices, const double* coefficients,
                              int face, int count, int faces,
                              const double* mask)
{
    const int* ix=indices+face;
    const double* co=coefficients+face;
    __m256d loaded;
    __m128i loadedIndices;
    if constexpr (Terminal) {
        const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                          _mm_setr_epi32(0,1,2,3));
        loadedIndices=_mm_maskload_epi32(ix,safe);
        loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
    }
    const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
    const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
    const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
    const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
    const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
    const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
    const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
    const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
    const __m128d p01=_mm_and_pd(_mm_mul_pd(x01,co01),_mm_loadu_pd(mask));
    const __m128d p23=_mm_and_pd(_mm_mul_pd(x23,co23),_mm_loadu_pd(mask+2));
    value-=_mm_cvtsd_f64(p01);
    value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
    value-=_mm_cvtsd_f64(p23);
    value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
    (void)count;
}

template<bool Terminal, bool Shared>
__attribute__((target("avx2,no-fma"), always_inline))
inline void optimizedRowsMaskedScatter(double* x, double* bPrime,
                         const int* starts, const int* indices,
                         const double* coefficients, const double* diag,
                         int first, int last, int faces,
                         const __m128i (*laneMasks)[4])
{
    for (int cell=first; cell<last; ++cell) {
        const int begin=starts[cell], end=starts[cell+1];
        const int degree=end-begin;
        const int tailCount=degree&3;
        const double* tailMask=reinterpret_cast<const double*>(activeMasks[tailCount]);
        double value=bPrime[cell];
        if constexpr (Shared) {
            const int fullEnd=end-tailCount;
            const double* fullMask=reinterpret_cast<const double*>(activeMasks[4]);
            int face=begin;
            for (; face<fullEnd; face+=4)
                maskedGatherChunk<false>(value,x,indices,coefficients,face,4,faces,fullMask);
            if (tailCount)
                maskedGatherChunk<Terminal>(value,x,indices,coefficients,
                    face,tailCount,faces,tailMask);
        } else {
            for (int face=begin; face<end; face+=4) {
                const int count=std::min(4,end-face);
                const int* ix=indices+face;
                const double* co=coefficients+face;
                __m256d loaded;
                __m128i loadedIndices;
                if constexpr (Terminal) {
                    const __m128i safe=_mm_cmpgt_epi32(_mm_set1_epi32(faces-face),
                                                      _mm_setr_epi32(0,1,2,3));
                    loadedIndices=_mm_maskload_epi32(ix,safe);
                    loaded=_mm256_maskload_pd(co,_mm256_cvtepi32_epi64(safe));
                }
                const int i0=Terminal?_mm_extract_epi32(loadedIndices,0):ix[0];
                const int i1=Terminal?_mm_extract_epi32(loadedIndices,1):ix[1];
                const int i2=Terminal?_mm_extract_epi32(loadedIndices,2):ix[2];
                const int i3=Terminal?_mm_extract_epi32(loadedIndices,3):ix[3];
                const double* mask=reinterpret_cast<const double*>(activeMasks[count]);
                const __m128d x01=_mm_setr_pd(x[i0],x[i1]);
                const __m128d x23=_mm_setr_pd(x[i2],x[i3]);
                const __m128d co01=Terminal?_mm256_castpd256_pd128(loaded):_mm_loadu_pd(co);
                const __m128d co23=Terminal?_mm256_extractf128_pd(loaded,1):_mm_loadu_pd(co+2);
                const __m128d p01=_mm_and_pd(_mm_mul_pd(x01,co01),_mm_loadu_pd(mask));
                const __m128d p23=_mm_and_pd(_mm_mul_pd(x23,co23),_mm_loadu_pd(mask+2));
                value-=_mm_cvtsd_f64(p01);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p01,p01));
                value-=_mm_cvtsd_f64(p23);
                value-=_mm_cvtsd_f64(_mm_unpackhi_pd(p23,p23));
            }
        }
        value/=diag[cell];
        const int fullEnd=end-tailCount;
        if constexpr (Shared) {
            int face=begin;
            for (; face<fullEnd; face+=4)
                maskedScatterChunk<false,true>(bPrime,indices,coefficients,
                    face,4,faces,value,laneMasks);
            if (tailCount)
                maskedScatterChunk<Terminal,true>(bPrime,indices,coefficients,
                    face,tailCount,faces,value,laneMasks);
        } else {
            int face=begin;
            for (; face<fullEnd; face+=4)
                maskedScatterChunk<false,false>(bPrime,indices,coefficients,
                    face,4,faces,value,laneMasks);
            if (tailCount)
                maskedScatterChunk<Terminal,false>(bPrime,indices,coefficients,
                    face,tailCount,faces,value,laneMasks);
        }
        x[cell]=value;
    }
}

template<bool Shared>
__attribute__((target("avx2,no-fma"), noinline))
inline void optimizedMinimal4MaskedScatter(Foam::scalarField& psi,
    const smootherTest::gamg::GAMGMatrix& matrix,
    const Foam::scalarField& source, int sweeps, MaskedScatterData& data)
{
    const int n=matrix.nCells(), faces=matrix.nFaces();
    const int* starts=matrix.ownerStartAddr().data();
    const int* indices=matrix.upperAddr().data();
    const double* coefficients=matrix.upper().data();
    const double* diag=matrix.diag().data();
    const double* rhs=source.data();
    double* x=psi.data();
    double* bPrime=data.base.bPrime.data();
    for (int sweep=0; sweep<sweeps; ++sweep) {
        std::copy(rhs,rhs+n,bPrime);
        optimizedRowsMaskedScatter<false,Shared>(x,bPrime,starts,indices,coefficients,
            diag,0,data.base.terminalRow,faces,data.laneMasks);
        optimizedRowsMaskedScatter<true,Shared>(x,bPrime,starts,indices,coefficients,
            diag,data.base.terminalRow,n,faces,data.laneMasks);
    }
}
}
