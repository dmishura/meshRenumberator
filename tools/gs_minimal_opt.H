#pragma once
#include "gs_minimal4.H"
#include <algorithm>
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
}
