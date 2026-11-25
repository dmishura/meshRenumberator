# M6.4 large-mesh performance summary

Original GS, 1 thread; seven samples, 64 measured sweeps/sample, eight warmups, CPU 0. Values are median ± sample SD in ms/sweep. Scalar and M6 are the archived matched baselines; M6.4a/b were measured in the timing-first campaign. No PMU profiling was run in this summary.

| Mesh / numbering | Scalar | M6 | M6.4a | Δ vs M6 | M6.4b | Δ vs M6 |
|---|---:|---:|---:|---:|---:|---:|
| WD / DamBreak / Original | 74.517 ± 0.158 | 77.690 ± 0.063 | 77.041 ± 0.172 | -0.84% | 77.042 ± 0.395 | -0.83% |
| WD / DamBreak / RCM | 33.539 ± 0.115 | 33.908 ± 0.065 | 33.601 ± 0.039 | -0.91% | 33.504 ± 0.065 | -1.19% |
| Cavity3d / Original | 63.171 ± 0.106 | 65.303 ± 0.081 | 64.592 ± 0.150 | -1.09% | 64.649 ± 0.086 | -1.00% |
| AHBody / RCM | 14.748 ± 0.056 | 15.305 ± 0.003 | 15.200 ± 0.055 | -0.69% | 15.150 ± 0.056 | -1.01% |
| MTBHPC / RCM | 63.091 ± 0.092 | 59.266 ± 0.055 | 59.518 ± 0.118 | +0.43% | 59.135 ± 0.134 | -0.22% |

Negative Δ means M6.4 is faster than M6. Relative speedup is `M6 / variant`; positive regression percentage corresponds to Δ above zero.

| Mesh / numbering | M6.4a speedup vs M6 | M6.4b speedup vs M6 | M6.4a speedup vs Scalar | M6.4b speedup vs Scalar |
|---|---:|---:|---:|---:|
| WD / DamBreak / Original | 1.0084× | 1.0084× | 0.9672× | 0.9672× |
| WD / DamBreak / RCM | 1.0091× | 1.0120× | 0.9982× | 1.0010× |
| Cavity3d / Original | 1.0110× | 1.0101× | 0.9780× | 0.9771× |
| AHBody / RCM | 1.0069× | 1.0102× | 0.9703× | 0.9735× |
| MTBHPC / RCM | 0.9958× | 1.0022× | 1.0600× | 1.0669× |

## Takeaway

M6.4a/b make small, mesh-dependent changes relative to M6. They are still slower than Scalar on WD_DamBreak Original, Cavity3d Original and AHBody RCM. On WD_DamBreak RCM they are effectively tied with Scalar within the observed spread. MTBHPC RCM remains the clear case where M6 is beneficial; M6.4b is marginally faster than M6. M6 remains the default pending broader evidence.
