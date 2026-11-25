#pragma once
#include <immintrin.h>

namespace gsExperiment {
// Scalar arithmetic and lane blends preserve the original reduction order.
__attribute__((target("avx2,no-fma"), always_inline))
inline double lane(double value, double coefficient, int neighbour,
                   long long active, const double* psi)
{
    const __m128d mask = _mm_castsi128_pd(_mm_set1_epi64x(active));
    const __m128d x = _mm_and_pd(_mm_load_sd(psi + neighbour), mask);
    const __m128d product = _mm_mul_sd(_mm_set_sd(coefficient), x);
    const __m128d previous = _mm_set_sd(value);
    const __m128d next = _mm_sub_sd(previous, product);
    return _mm_cvtsd_f64(_mm_blendv_pd(previous, next, mask));
}

// Both input streams use masked loads, including the final partial window.
template<bool Window>
__attribute__((target("avx2,no-fma"), noinline))
void smooth(Foam::scalarField& psi, const smootherTest::gamg::GAMGMatrix& matrix,
            const Foam::scalarField& source, int sweeps,
            Foam::scalarField* workspace = nullptr)
{
    Foam::scalarField local;
    if (!workspace) local.resize(source.size());
    Foam::scalarField& bPrime = workspace ? *workspace : local;
    const auto& starts = matrix.ownerStartAddr();
    const auto* coefficients = matrix.upper().data();
    const auto* indices = matrix.upperAddr().data();
    const int faces = matrix.upper().size();
    for (int sweep=0; sweep<sweeps; ++sweep)
    {
        bPrime = source;
        __m256d buffer = _mm256_setzero_pd();
        __m128i indexBuffer = _mm_setzero_si128();
        int position = 0;
        if constexpr (Window)
        {
            const __m128i mask = _mm_cmpgt_epi32(_mm_set1_epi32(faces), _mm_setr_epi32(0,1,2,3));
            indexBuffer = _mm_maskload_epi32(indices, mask);
            buffer = _mm256_maskload_pd(coefficients, _mm256_cvtepi32_epi64(mask));
        }
        for (int cell=0; cell<matrix.nCells(); ++cell)
        {
            double value = bPrime[cell];
            const int end = starts[cell+1];
            for (int face=starts[cell]; face<end; face+=4)
            {
                const int count = std::min(4, end-face);
                const __m128i valid = _mm_cmpgt_epi32(_mm_set1_epi32(count), _mm_setr_epi32(0,1,2,3));
                __m128i ix;
                __m256d co;
                if constexpr (Window) { ix=indexBuffer; co=buffer; }
                else
                {
                    ix = _mm_maskload_epi32(indices+face, valid);
                    co = _mm256_maskload_pd(coefficients+face, _mm256_cvtepi32_epi64(valid));
                }
                ix = _mm_and_si128(ix, valid);
                co = _mm256_and_pd(co, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(valid)));
                const __m128d lo = _mm256_castpd256_pd128(co);
                const __m128d hi = _mm256_extractf128_pd(co,1);
                value=lane(value,_mm_cvtsd_f64(lo),_mm_extract_epi32(ix,0),-(long long)(count>0),psi.data());
                value=lane(value,_mm_cvtsd_f64(_mm_unpackhi_pd(lo,lo)),_mm_extract_epi32(ix,1),-(long long)(count>1),psi.data());
                value=lane(value,_mm_cvtsd_f64(hi),_mm_extract_epi32(ix,2),-(long long)(count>2),psi.data());
                value=lane(value,_mm_cvtsd_f64(_mm_unpackhi_pd(hi,hi)),_mm_extract_epi32(ix,3),-(long long)(count>3),psi.data());
                if constexpr (Window)
                {
                    // Preserve unused lanes and refill the newly opened tail.
                    const int next = position+count;
                    const __m128i safe = _mm_cmpgt_epi32(_mm_set1_epi32(faces-next),_mm_setr_epi32(0,1,2,3));
                    const __m128i tail = _mm_cmpgt_epi32(_mm_setr_epi32(0,1,2,3),_mm_set1_epi32(3-count));
                    const __m128i refill = _mm_and_si128(safe,tail);
                    const __m256i shift = _mm256_setr_epi32(2*count,2*count+1,2*count+2,2*count+3,2*count+4,2*count+5,2*count+6,2*count+7);
                    const __m256d shifted = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(buffer),shift));
                    const __m128i ishift = _mm_setr_epi32(count&3,(count+1)&3,(count+2)&3,(count+3)&3);
                    const __m128i shiftedIx = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(indexBuffer),_mm256_castsi128_si256(ishift)));
                    const __m128i freshIx = _mm_maskload_epi32(indices+next,refill);
                    const __m256d fresh = _mm256_maskload_pd(coefficients+next,_mm256_cvtepi32_epi64(refill));
                    buffer = _mm256_blendv_pd(shifted,fresh,_mm256_castsi256_pd(_mm256_cvtepi32_epi64(tail)));
                    indexBuffer = _mm_blendv_epi8(shiftedIx,freshIx,tail);
                    position=next;
                }
            }
            value /= matrix.diag()[cell];
            for (int face=starts[cell]; face<end; ++face)
                bPrime[indices[face]] -= coefficients[face]*value;
            psi[cell]=value;
        }
    }
}
}
