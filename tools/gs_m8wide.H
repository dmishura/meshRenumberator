#pragma once

#include "gs_m74d.H"
#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace gsExperiment {

// Tail masks are indexed by the number of inactive elements in the final
// eight-element chunk.  Rows passed to the wide path always have at least
// eight allocated face entries from any chunk start; the final seven face
// entries of the complete LDU stream are delegated to the existing safe M6.4d
// terminal helper.
alignas(64) inline constexpr std::uint16_t m8TailMasks128[8][4] = {
    {3,3,3,3}, {3,3,3,1}, {3,3,3,0}, {3,3,1,0},
    {3,3,0,0}, {3,1,0,0}, {3,0,0,0}, {1,0,0,0}
};
alignas(64) inline constexpr std::uint8_t m8TailMasks256[8][2] = {
    {15,15}, {15,7}, {15,3}, {15,1}, {15,0}, {7,0}, {3,0}, {1,0}
};
alignas(64) inline constexpr std::uint8_t m8TailMasks512[8] = {
    255,127,63,31,15,7,3,1
};

inline int m8TerminalRow8(const smootherTest::gamg::GAMGMatrix& matrix)
{
    const int faces = matrix.nFaces();
    if (faces < 8) return 0;
    const int* starts = matrix.ownerStartAddr().data();
    const int n = matrix.nCells();
    return static_cast<int>(std::upper_bound(starts + 1, starts + n + 1,
                                              faces - 7) - starts - 1);
}

template<int Variant>
__attribute__((target("avx512f,avx512vl,avx512dq,no-fma"), always_inline))
inline double m8WideChunk(double value, int face, int end,
    const int* indices, const double* psi, const double* coefficients,
    const void* maskBase)
{
    static_assert(Variant >= 0 && Variant <= 2);
    const std::uint64_t faceIndex = static_cast<std::uint64_t>(face);
    register const std::int32_t zero asm("r12") = 0;

    if constexpr (Variant == 0) {
        // Four independent XMM products, with each product covering two
        // consecutive slots.  The mask table contains one k-register value
        // per pair and tail position.
        std::int64_t tmp;
        asm volatile(
            "leal 8(%k[face]), %k[tmp]\n\t"
            "subl %k[end], %k[tmp]\n\t"
            "cmovs %k[zero], %k[tmp]\n\t"
            "kmovw 0(%[mask],%[tmp],8), %%k1\n\t"
            "kmovw 2(%[mask],%[tmp],8), %%k2\n\t"
            "kmovw 4(%[mask],%[tmp],8), %%k3\n\t"
            "kmovw 6(%[mask],%[tmp],8), %%k4\n\t"
            "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm0\n\t"
            "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm0, %%xmm0\n\t"
            "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm1\n\t"
            "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm1, %%xmm1\n\t"
            "movslq 16(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm2\n\t"
            "movslq 20(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm2, %%xmm2\n\t"
            "movslq 24(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm3\n\t"
            "movslq 28(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm3, %%xmm3\n\t"
            "vmulpd 0(%[co],%[face],8), %%xmm0, %%xmm0%{%%k1%}%{z%}\n\t"
            "vmulpd 16(%[co],%[face],8), %%xmm1, %%xmm1%{%%k2%}%{z%}\n\t"
            "vmulpd 32(%[co],%[face],8), %%xmm2, %%xmm2%{%%k3%}%{z%}\n\t"
            "vmulpd 48(%[co],%[face],8), %%xmm3, %%xmm3%{%%k4%}%{z%}\n\t"
            "vsubsd %%xmm0, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm0, %%xmm0, %%xmm0\n\t"
            "vsubsd %%xmm0, %[acc], %[acc]\n\t"
            "vsubsd %%xmm1, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm1, %%xmm1, %%xmm1\n\t"
            "vsubsd %%xmm1, %[acc], %[acc]\n\t"
            "vsubsd %%xmm2, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm2, %%xmm2, %%xmm2\n\t"
            "vsubsd %%xmm2, %[acc], %[acc]\n\t"
            "vsubsd %%xmm3, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm3, %%xmm3, %%xmm3\n\t"
            "vsubsd %%xmm3, %[acc], %[acc]"
            : [acc] "+x"(value), [tmp] "=&r"(tmp)
            : [ix] "r"(indices), [face] "r"(faceIndex), [psi] "r"(psi),
              [co] "r"(coefficients), [end] "r"(end), [zero] "r"(zero),
              [mask] "r"(maskBase)
            : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3",
              "k1", "k2", "k3", "k4");
    } else if constexpr (Variant == 1) {
        // Two independent YMM products.  The scalar psi reads are packed as
        // pairs first; the two 4-lane masks are loaded directly into k1/k2.
        std::int64_t tmp;
        asm volatile(
            "leal 8(%k[face]), %k[tmp]\n\t"
            "subl %k[end], %k[tmp]\n\t"
            "cmovs %k[zero], %k[tmp]\n\t"
            "kmovb 0(%[mask],%[tmp],2), %%k1\n\t"
            "kmovb 1(%[mask],%[tmp],2), %%k2\n\t"
            "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm4\n\t"
            "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm4, %%xmm4\n\t"
            "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm5\n\t"
            "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm5, %%xmm5\n\t"
            "vinsertf128 $1, %%xmm5, %%ymm4, %%ymm4\n\t"
            "movslq 16(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm6\n\t"
            "movslq 20(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm6, %%xmm6\n\t"
            "movslq 24(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm7\n\t"
            "movslq 28(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm7, %%xmm7\n\t"
            "vinsertf128 $1, %%xmm7, %%ymm6, %%ymm6\n\t"
            "vmulpd 0(%[co],%[face],8), %%ymm4, %%ymm4%{%%k1%}%{z%}\n\t"
            "vmulpd 32(%[co],%[face],8), %%ymm6, %%ymm6%{%%k2%}%{z%}\n\t"
            "vextractf128 $1, %%ymm4, %%xmm5\n\t"
            "vextractf128 $1, %%ymm6, %%xmm7\n\t"
            "vsubsd %%xmm4, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm4, %%xmm4, %%xmm4\n\t"
            "vsubsd %%xmm4, %[acc], %[acc]\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm5, %%xmm5, %%xmm5\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            "vsubsd %%xmm6, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm6, %%xmm6, %%xmm6\n\t"
            "vsubsd %%xmm6, %[acc], %[acc]\n\t"
            "vsubsd %%xmm7, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm7, %%xmm7, %%xmm7\n\t"
            "vsubsd %%xmm7, %[acc], %[acc]"
            : [acc] "+x"(value), [tmp] "=&r"(tmp)
            : [ix] "r"(indices), [face] "r"(faceIndex), [psi] "r"(psi),
              [co] "r"(coefficients), [end] "r"(end), [zero] "r"(zero),
              [mask] "r"(maskBase)
            : "cc", "memory", "ymm4", "ymm5", "ymm6", "ymm7",
              "k1", "k2");
    } else {
        // One ZMM product.  Three 128-bit sub-vectors are extracted after
        // multiplication so that lane contributions can still be subtracted
        // one by one in the original scalar order.
        std::int64_t tmp;
        asm volatile(
            "leal 8(%k[face]), %k[tmp]\n\t"
            "subl %k[end], %k[tmp]\n\t"
            "cmovs %k[zero], %k[tmp]\n\t"
            "kmovb 0(%[mask],%[tmp],1), %%k1\n\t"
            "movslq 0(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm4\n\t"
            "movslq 4(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm4, %%xmm4\n\t"
            "movslq 8(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm5\n\t"
            "movslq 12(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm5, %%xmm5\n\t"
            "vinsertf128 $1, %%xmm5, %%ymm4, %%ymm4\n\t"
            "movslq 16(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm6\n\t"
            "movslq 20(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm6, %%xmm6\n\t"
            "movslq 24(%[ix],%[face],4), %[tmp]\n\t"
            "vmovsd (%[psi],%[tmp],8), %%xmm7\n\t"
            "movslq 28(%[ix],%[face],4), %[tmp]\n\t"
            "vmovhpd (%[psi],%[tmp],8), %%xmm7, %%xmm7\n\t"
            "vinsertf128 $1, %%xmm7, %%ymm6, %%ymm6\n\t"
            "vmovaps %%ymm4, %%ymm16\n\t"
            "vinsertf64x4 $1, %%ymm6, %%zmm16, %%zmm16\n\t"
            "vmulpd 0(%[co],%[face],8), %%zmm16, %%zmm16%{%%k1%}%{z%}\n\t"
            "vextractf64x2 $1, %%zmm16, %%xmm5\n\t"
            "vextractf64x2 $2, %%zmm16, %%xmm6\n\t"
            "vextractf64x2 $3, %%zmm16, %%xmm7\n\t"
            "vsubsd %%xmm16, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm16, %%xmm16, %%xmm16\n\t"
            "vsubsd %%xmm16, %[acc], %[acc]\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm5, %%xmm5, %%xmm5\n\t"
            "vsubsd %%xmm5, %[acc], %[acc]\n\t"
            "vsubsd %%xmm6, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm6, %%xmm6, %%xmm6\n\t"
            "vsubsd %%xmm6, %[acc], %[acc]\n\t"
            "vsubsd %%xmm7, %[acc], %[acc]\n\t"
            "vunpckhpd %%xmm7, %%xmm7, %%xmm7\n\t"
            "vsubsd %%xmm7, %[acc], %[acc]"
            : [acc] "+x"(value), [tmp] "=&r"(tmp)
            : [ix] "r"(indices), [face] "r"(faceIndex), [psi] "r"(psi),
              [co] "r"(coefficients), [end] "r"(end), [zero] "r"(zero),
              [mask] "r"(maskBase)
            : "cc", "memory", "zmm4", "zmm5", "zmm6", "zmm7",
              "zmm16", "k1");
    }
    return value;
}

template<int Variant>
__attribute__((target("avx512f,avx512vl,avx512dq,no-fma"), optimize("fp-contract=off"), noinline))
inline void minimal8Avx512(Foam::scalarField& psi,
    const smootherTest::gamg::GAMGMatrix& matrix,
    const Foam::scalarField& source, int sweeps, MinimalOptData& data)
{
    const int n = matrix.nCells();
    const int faces = matrix.nFaces();
    const int* starts = matrix.ownerStartAddr().data();
    const int* indices = matrix.upperAddr().data();
    const double* coefficients = matrix.upper().data();
    const double* diag = matrix.diag().data();
    const double* rhs = source.data();
    double* x = psi.data();
    double* bPrime = data.bPrime.data();
    const void* maskAddress;
    if constexpr (Variant == 0) maskAddress = m8TailMasks128;
    else if constexpr (Variant == 1) maskAddress = m8TailMasks256;
    else maskAddress = m8TailMasks512;
    register const void* maskBase asm("r15") = maskAddress;

    // Keep all eight scalar index/psi loads in-bounds on the main path.  Rows
    // that can begin inside the final seven physical face slots use the
    // existing masked-load M6.4d terminal path instead.
    const int terminalRow8 = m8TerminalRow8(matrix);

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        std::copy(rhs, rhs + n, bPrime);
        for (int cell = 0; cell < terminalRow8; ++cell) {
            const int begin = starts[cell];
            const int end = starts[cell + 1];
            double value = bPrime[cell];
            for (int face = begin; face < end; face += 8)
                value = m8WideChunk<Variant>(value, face, end, indices, x,
                                             coefficients, maskBase);
            value /= diag[cell];
            for (int face = begin; face < end; ++face)
                bPrime[indices[face]] -= coefficients[face] * value;
            x[cell] = value;
        }
        optimizedTerminalRowsIsolated(x, bPrime, starts, indices,
            coefficients, diag, terminalRow8, n, faces);
    }
}

} // namespace gsExperiment
