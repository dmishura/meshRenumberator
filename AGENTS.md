# Repository Guidelines

## Project Structure & Module Organization

This workspace directory is the mesh-renumbering project area. The related benchmark repository is in the sibling `../WavefrontGaussSeidel` directory: C++ kernels and harness code are under `src/` and `tests/`, while mesh readers, ordering analysis, benchmarking, and visualization utilities are under `tools/`. Put new renumbering code in `tools/` and its focused tests in `tools/tests/`. Keep generated meshes, reports, and experiment artifacts out of source directories.

## Build, Test, and Development Commands

From `../WavefrontGaussSeidel`, configure and build with `cmake -S . -B build-cmake && cmake --build build-cmake -j`. Run the C++ and Python correctness suite with `ctest --test-dir build-cmake -L correctness --output-on-failure`; the Makefile also offers `make test-modern`. Run a targeted Python test with `python3 -m unittest discover -s tools/tests -p test_openfoam_mesh_reader.py`. Use the existing benchmark harness and its saved plans for performance work; keep benchmark comparisons matched for mesh, build, affinity, and thread count.

## Coding Style & Naming Conventions

Follow neighboring code. Python utilities use four-space indentation, type hints, descriptive `snake_case` names, and standard-library modules where practical. C++ code follows the project’s existing formatting and naming in `src/` and `tests/`. Prefer extending `openfoam_mesh_reader.py`, `ih_permutation.py`, and the benchmark utilities over duplicating their formats or logic.

## Testing Guidelines

Add focused tests under `tools/tests/` for Python behavior and under `tests/` for C++ behavior. Use `test_*.py` names and `unittest`; register build-level tests through the existing CMake/CTest setup when appropriate. Validate mesh topology and permutation invariants before interpreting any performance measurements.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries (for example, “Extend tools with new locality analysis stuff”). Keep commits focused and describe the behavior changed. A pull request should explain the mesh format and algorithm choices, list validation and benchmark commands, include representative results when performance is involved, and identify generated artifacts or unsupported input features. Preserve unrelated working-tree changes.

## Mesh and Benchmark Safety

Never overwrite source meshes. Write each generated mesh and its permutation, validation report, and metrics into a unique experiment directory. Reject unsupported OpenFOAM encodings or metadata explicitly; do not silently drop boundary or face semantics. Run correctness validation before benchmarking and avoid simultaneous competing benchmark runs.
