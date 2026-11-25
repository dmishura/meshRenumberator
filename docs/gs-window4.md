# Four-slot Original GS experiment

The existing `tools/gs_probe.C` supports `MESH_GS_VARIANT=SWndReload4` and `MESH_GS_VARIANT=SWnd4`. The experimental gather code is in `tools/gs_window4.H`. Build using `bash tools/build_probe.sh experiments/window4/gs_probe` and use the established source mesh, output mesh, cell map, face map, thread, sweep, repetition and width arguments.

```bash
MESH_GS_VARIANT=SWndReload4 taskset -c 0 experiments/window4/gs_probe \
  SOURCE_POLYMESH VARIANT_POLYMESH cells.bin faces.bin 1 64 7 1024
MESH_GS_VARIANT=SWnd4 taskset -c 0 experiments/window4/gs_probe \
  SOURCE_POLYMESH VARIANT_POLYMESH cells.bin faces.bin 1 64 7 1024
```

Before measuring, the probe checks bitwise equality after one and 192 sweeps against Original GS for the same numbering. The probe prepares persistent bPrime storage before experimental validation and measurements. Eight warmup sweeps precede measurement through the existing measurement helper.

Run focused synthetic tests with the existing probe build helper.

```bash
bash tools/build_probe.sh experiments/window4/test tools/tests/gs_window4_test.C
experiments/window4/test
```

The PMU collector accepts both names through `--kernel`. It uses the existing FIFO interval and excludes validation and warmup from collection. Run experiments serially. These kernels require AVX2 support. The target attribute applies AVX2 only to the experimental functions.

Reload4 masked-loads each four-face chunk. Window4 keeps coefficients and indices across rows, shifts unused elements and masked-loads the opened tail. Psi values are loaded when the current row is processed. Inactive indices select psi element zero and their loaded bits are masked before multiplication. Accumulator blends preserve the previous value for inactive slots. Real face contributions retain the sequential scalar reduction order.

## MinimalReload4 and Padded4

`tools/gs_minimal4.H` adds prepared gather data and persistent bPrime storage. Select either variant through `MESH_GS_VARIANT=MinimalReload4` or `MESH_GS_VARIANT=Padded4`. The PMU collector accepts the same names through `--kernel`.

MinimalReload4 copies the original coefficient/index streams and appends three initialized slots. It loads four faces unconditionally, multiplies once with AVX2, masks products using a five-entry constant table, and performs four scalar subtractions in face order. Inactive multiplications may set floating-point exception flags before their products are masked. This experiment assumes ordinary round-to-nearest benchmark inputs and does not preserve all exceptional IEEE-754 behavior.

Padded4 preserves real entries in each row and pads the row to a multiple of four. Dummy entries use coefficient +0 and the allocated psi sentinel at index nCells. The sentinel remains +0 throughout the solve. Gather uses four explicit scalar multiply/subtract statements. Scatter reads the same padded coefficient/index arrays, starting at paddedStarts[cell] and processing only the original number of real faces. Dummy entries are skipped and the sentinel remains untouched.

Construction, allocation and padding precede measured sweeps. The probe's first warmup also allocates the psi sentinel. Both variants check bitwise equality after one and 192 sweeps against the established scalar path. Synthetic tests are in `tools/tests/gs_window4_test.C`.

To run only experimental correctness checks, set `MESH_GS_VALIDATE_ONLY=1` together with the selected `MESH_GS_VARIANT`. The probe exits after its bitwise one-sweep and 192-sweep checks and sentinel validation.

## Optimized MinimalReload4

`tools/gs_minimal_opt.H` implements `gsExperiment::optimizedMinimal4`. Select `MESH_GS_VARIANT=OptimizedMinimalReload4`, or use `--kernel OptimizedMinimalReload4` in the PMU collector. The reference `MinimalReload4` remains available.

The optimized kernel borrows coefficient, index and row-start pointers directly from GAMGMatrix. Gather and scatter use those same arrays and the same row interval. Matrix storage overhead is constant. Its only numerical heap storage is the conventional bPrime workspace. Before solving, a binary search identifies the first row ending beyond `nFaces-3`. Earlier rows permit ordinary four-slot loads entirely within the original allocation. A separately compiled terminal row range uses masked loads. No terminal-range branch occurs inside the main row or chunk loops.

Two XMM products feed four sequential scalar subtractions. The original aligned mask table is retained. Overlapping mask addressing, unsigned-index/restrict changes and pointer induction were evaluated and discarded. Intermediate implementations are archived under `experiments/minimal-opt-20261125/`, with timing samples and machine code.

The inactive-product floating-point exception restriction of MinimalReload4 also applies here. Normal round-to-nearest real benchmark inputs are verified bitwise after one and 192 sweeps. Run correctness only with `MESH_GS_VALIDATE_ONLY=1`.
