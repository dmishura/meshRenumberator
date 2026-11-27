#pragma once
#include "gs_minimal_opt.H"
#include <algorithm>
#include <immintrin.h>

namespace gsExperiment {
// Flat table includes one initialized trailing element for the final m128 load.
alignas(32) static constexpr long long scalarProductMasks[21]={
    0,0,0,0, -1,0,0,0, -1,-1,0,0, -1,-1,-1,0, -1,-1,-1,-1, 0
};

// Ordinary loads are legal for rows ending at or before nFaces-3.
// A separate terminal range uses masked loads from the same original arrays.
// This preserves matrix sizes and needs no copied LDU storage or tail allocation.
template<bool Terminal>
__attribute__((target("avx2,no-fma"), always_inline))
inline void scalarProductRows(double* x, double* bPrime, const int* starts,
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
            const double* mask=reinterpret_cast<const double*>(scalarProductMasks+4*count);
            const __m128d lo=Terminal?_mm256_castpd256_pd128(loaded):_mm_setzero_pd();
            const __m128d hi=Terminal?_mm256_extractf128_pd(loaded,1):_mm_setzero_pd();
            const __m128d c0=Terminal?lo:_mm_load_sd(co);
            const __m128d c1=Terminal?_mm_unpackhi_pd(lo,lo):_mm_load_sd(co+1);
            const __m128d c2=Terminal?hi:_mm_load_sd(co+2);
            const __m128d c3=Terminal?_mm_unpackhi_pd(hi,hi):_mm_load_sd(co+3);
            __m128d p0=_mm_load_sd(x+i0);
            __m128d p1=_mm_load_sd(x+i1);
            __m128d p2=_mm_load_sd(x+i2);
            __m128d p3=_mm_load_sd(x+i3);
            if constexpr (Terminal) {
                p0=_mm_mul_sd(p0,c0);
                p1=_mm_mul_sd(p1,c1);
                p2=_mm_mul_sd(p2,c2);
                p3=_mm_mul_sd(p3,c3);
            } else {
                // GCC retains four redundant coefficient loads with mul_sd intrinsics.
                asm("vmulsd %1,%0,%0" : "+x"(p0) : "m"(co[0]));
                asm("vmulsd %1,%0,%0" : "+x"(p1) : "m"(co[1]));
                asm("vmulsd %1,%0,%0" : "+x"(p2) : "m"(co[2]));
                asm("vmulsd %1,%0,%0" : "+x"(p3) : "m"(co[3]));
            }
            p0=_mm_and_pd(p0,_mm_loadu_pd(mask+0));
            p1=_mm_and_pd(p1,_mm_loadu_pd(mask+1));
            p2=_mm_and_pd(p2,_mm_loadu_pd(mask+2));
            p3=_mm_and_pd(p3,_mm_loadu_pd(mask+3));
            value-=_mm_cvtsd_f64(p0);
            value-=_mm_cvtsd_f64(p1);
            value-=_mm_cvtsd_f64(p2);
            value-=_mm_cvtsd_f64(p3);
        }
        value/=diag[cell];
        for (int face=begin; face<end; ++face)
            bPrime[indices[face]]-=coefficients[face]*value;
        x[cell]=value;
    }
}

__attribute__((target("avx2,no-fma"), noinline))
inline void scalarProductMinimal4(Foam::scalarField& psi,
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
        scalarProductRows<false>(x,bPrime,starts,indices,coefficients,diag,0,data.terminalRow,faces);
        scalarProductRows<true>(x,bPrime,starts,indices,coefficients,diag,data.terminalRow,n,faces);
    }
}
}
