#pragma once
#include <algorithm>
#include <chrono>
#include <immintrin.h>

namespace gsExperiment {
alignas(32) static constexpr long long activeMasks[5][4] = {
    {0,0,0,0}, {-1,0,0,0}, {-1,-1,0,0}, {-1,-1,-1,0}, {-1,-1,-1,-1}
};

// Prepared gather storage and persistent workspace. Construction precedes timing.
struct Minimal4Data {
    Foam::labelField starts, indices;
    Foam::scalarField coefficients, bPrime;
    bool padded;
    double preprocessingMs;
    Minimal4Data(const smootherTest::gamg::GAMGMatrix& matrix, bool rowPadding)
      : bPrime(matrix.nCells()), padded(rowPadding)
    {
        const auto begin = std::chrono::steady_clock::now();
        if (padded) {
            starts.push_back(0);
            for (int cell=0; cell<matrix.nCells(); ++cell) {
                const int first=matrix.ownerStartAddr()[cell];
                const int last=matrix.ownerStartAddr()[cell+1];
                for (int face=first; face<last; ++face) {
                    indices.push_back(matrix.upperAddr()[face]);
                    coefficients.push_back(matrix.upper()[face]);
                }
                for (int pad=0; pad<((4-(last-first)%4)%4); ++pad) {
                    indices.push_back(matrix.nCells());
                    coefficients.push_back(0.0);
                }
                starts.push_back(indices.size());
            }
        } else {
            starts=matrix.ownerStartAddr();
            indices=matrix.upperAddr();
            coefficients=matrix.upper();
            indices.resize(indices.size()+3,0);
            coefficients.resize(coefficients.size()+3,0.0);
        }
        preprocessingMs=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-begin).count();
    }
};

template<bool Padded>
__attribute__((target("avx2,no-fma"), noinline))
void minimal4(Foam::scalarField& psi, const smootherTest::gamg::GAMGMatrix& matrix,
              const Foam::scalarField& source, int sweeps, Minimal4Data& data)
{
    const int* starts=data.starts.data();
    const int* indices=data.indices.data();
    const double* coefficients=data.coefficients.data();
    double* bPrime=data.bPrime.data();
    for (int sweep=0; sweep<sweeps; ++sweep) {
        std::copy(source.begin(),source.end(),bPrime);
        for (int cell=0; cell<matrix.nCells(); ++cell) {
            double value=bPrime[cell];
            const int end=starts[cell+1];
            for (int face=starts[cell]; face<end; face+=4) {
                if constexpr (Padded) {
                    value-=coefficients[face]*psi[indices[face]];
                    value-=coefficients[face+1]*psi[indices[face+1]];
                    value-=coefficients[face+2]*psi[indices[face+2]];
                    value-=coefficients[face+3]*psi[indices[face+3]];
                } else {
                    const int count=std::min(4,end-face);
                    const __m256d x=_mm256_setr_pd(psi[indices[face]],psi[indices[face+1]],
                                                   psi[indices[face+2]],psi[indices[face+3]]);
                    const __m256d mask=_mm256_castsi256_pd(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(activeMasks[count])));
                    const __m256d product=_mm256_and_pd(
                        _mm256_mul_pd(_mm256_loadu_pd(coefficients+face),x),mask);
                    const __m128d lo=_mm256_castpd256_pd128(product);
                    const __m128d hi=_mm256_extractf128_pd(product,1);
                    value-=_mm_cvtsd_f64(lo);
                    value-=_mm_cvtsd_f64(_mm_unpackhi_pd(lo,lo));
                    value-=_mm_cvtsd_f64(hi);
                    value-=_mm_cvtsd_f64(_mm_unpackhi_pd(hi,hi));
                }
            }
            value/=matrix.diag()[cell];
            if constexpr (Padded) {
                // Real entries precede padding in each row. Skip dummy slots,
                // which reference psi's sentinel outside the bPrime allocation.
                const int realLength=matrix.ownerStartAddr()[cell+1]
                                   -matrix.ownerStartAddr()[cell];
                const int realEnd=starts[cell]+realLength;
                for (int face=starts[cell]; face<realEnd; ++face)
                    bPrime[indices[face]]-=coefficients[face]*value;
            } else {
                for (int face=matrix.ownerStartAddr()[cell]; face<matrix.ownerStartAddr()[cell+1]; ++face)
                    bPrime[matrix.upperAddr()[face]]-=matrix.upper()[face]*value;
            }
            psi[cell]=value;
        }
    }
}
}
