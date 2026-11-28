// Adapter for the existing GAMG L0 GS and IH1 kernels.
#include "GAMGKernels.H"
#include "ScheduleBuilder.H"
#include "fwMatrix.H"
#include "gs_window4.H"
#include "gs_minimal4.H"
#include "gs_minimal_opt.H"
#include "gs_m7.H"
#include "gs_m74d.H"
#include "gs_m8.H"
#include "gs_avx512.H"
#include "gs_m8wide.H"
#include <cstring>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace wt = smootherTest;
namespace gamg = wt::gamg;
namespace fw = wt::fwgamg;
namespace harness = wt::harness;
using Clock = std::chrono::steady_clock;
using Foam::scalarField;

void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Foam::labelField readMap(const char* path, std::size_t size)
{
    Foam::labelField result(size);
    std::ifstream input(path, std::ios::binary);
    for (auto& value : result)
    {
        unsigned char bytes[4];
        check(bool(input.read(reinterpret_cast<char*>(bytes), 4)), "truncated map");
        const auto decoded = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
            | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
        check(decoded < size, "map entry out of range");
        value = static_cast<Foam::label>(decoded);
    }
    check(input.peek() == EOF, "trailing map data");
    return result;
}

scalarField gather(const std::vector<double>& values, const std::vector<int>& order)
{
    scalarField result(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) result[i] = values[order[i]];
    return result;
}

double maxError(const std::vector<double>& a, const std::vector<double>& b)
{
    double error = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite solution");
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    return error;
}

double millis(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Opt-in perf stat control. The PMU stays disabled for mesh setup,
// validation, warmup, and psi resets. This only instruments the driver.
class PerfControl
{
    std::ofstream control_;
    std::ifstream ack_;

public:
    PerfControl(const char* control, const char* ack)
        : control_(control), ack_(ack)
    {
        check(bool(control_) && bool(ack_), "cannot open perf control FIFOs");
    }

    void send(const char* action)
    {
        control_ << action << std::endl;
        std::string reply;
        check(bool(control_) && bool(std::getline(ack_, reply)), "missing perf acknowledgement");
        // perf versions that write the terminating NUL leave it before
        // the next reply when the stream is read one line at a time.
        reply.erase(std::remove(reply.begin(), reply.end(), '\0'), reply.end());
        check(reply == "ack", "unexpected perf acknowledgement");
    }
};

template<class Solve>
void measure(const std::string& id, int threads, const scalarField& source,
             Solve solve, int sweeps, int repetitions, double setup, PerfControl* perf = nullptr)
{
    omp_set_num_threads(threads);
    int actual = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual = omp_get_num_threads();
    }
    check(actual == threads, "OpenMP team size differs from requested size");

    scalarField psi(source.size(), 0.0);
    // Warm the matrix and solver workspace outside the measured interval.
    solve(psi, 8);
    for (int i = 0; i < repetitions; ++i)
    {
        std::fill(psi.begin(), psi.end(), 0.0);
        if (perf) perf->send("enable");
        const auto start = Clock::now();
        solve(psi, sweeps);
        const double elapsed = millis(start);
        if (perf) perf->send("disable");
        check(std::isfinite(psi.front()), "nonfinite timed result");
        std::cout << "sample," << id << ',' << threads << ',' << i
                  << ',' << elapsed / sweeps << ',' << setup << '\n';
    }
}

int main(int argc, char** argv)
try
{
    check(argc >= 9, "usage: probe old new cell-map face-map threads sweeps repeats widths...");
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    const int threads = std::stoi(argv[5]);
    const int sweeps = std::stoi(argv[6]);
    const int repetitions = std::stoi(argv[7]);
    check(threads > 0 && sweeps > 0 && repetitions > 0, "expected positive counts");
    std::cout << std::setprecision(17);
    const auto loadStart = Clock::now();
    const auto original = wt::PolyMeshReader::read(argv[1]);
    const auto output = wt::PolyMeshReader::read(argv[2]);
    const auto cells = readMap(argv[3], original.nCells);
    const auto faces = readMap(argv[4], original.nInternalFaces());
    check(output.nCells == original.nCells &&
          output.nInternalFaces() == original.nInternalFaces(), "mesh size changed");
    const auto sourceMatrix = harness::makeMatrix(original);
    scalarField exact(original.nCells);
    for (std::size_t i = 0; i < exact.size(); ++i)
        exact[i] = 1.0 + 0.25 * std::sin(0.019 * i);
    const auto source = gather(harness::multiply(sourceMatrix, exact), cells);
    const auto expected = gather(exact, cells);
    const auto originalLdu = gamg::GAMGMatrix::fromLduMatrix(sourceMatrix);
    fw::NativeMatrixData native{
        static_cast<Foam::label>(output.nCells),
        Foam::labelField(output.owner.begin(), output.owner.begin() + faces.size()),
        Foam::labelField(output.neighbour.begin(), output.neighbour.end()),
        gather(sourceMatrix.diag(), cells), gather(sourceMatrix.upper(), faces)
    };
    // The harness matrix is symmetric, so exchanging the endpoints preserves
    // the coefficient. Its other possible value is checked explicitly.
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        const auto face = faces[i];
        const auto a = cells[output.owner[i]];
        const auto b = cells[output.neighbour[i]];
        check(std::min(a, b) == original.owner[face] &&
              std::max(a, b) == original.neighbour[face], "coefficient face mismatch");
        check(native.upper[i] == sourceMatrix.lower()[face], "matrix is not symmetric");
    }
    gamg::GAMGMatrix ldu(native.nCells, native.lowerAddr, native.upperAddr,
                        native.diag, native.upper);
    const double loadMs = millis(loadStart);
    const double centreError = std::max({
        maxError(output.cellCentreX, gather(original.cellCentreX, cells)),
        maxError(output.cellCentreY, gather(original.cellCentreY, cells)),
        maxError(output.cellCentreZ, gather(original.cellCentreZ, cells))
    });
    check(centreError < 1e-8, "cell centre correspondence failed");
    const auto gs = [&](scalarField& psi, int count)
    {
        gamg::gaussSeidelSmooth(psi, ldu, source, count);
    };
    scalarField reference(output.nCells, 0.0), converged(reference);
    scalarField originalSolution(reference), originalShort(reference);
    const auto originalSource = harness::multiply(sourceMatrix, exact);
    gamg::gaussSeidelSmooth(originalShort, originalLdu, originalSource, 1);
    gamg::gaussSeidelSmooth(originalSolution, originalLdu, originalSource, 192);
    gs(reference, 1);
    gs(converged, 192);
    const auto solutionError = maxError(expected, converged);
    const auto crossError = maxError(gather(originalSolution, cells), converged);
    const auto oneSweepDelta = maxError(gather(originalShort, cells), reference);
    check(solutionError < 1e-11 && crossError < 1e-11, "common GS solution mismatch");
    scalarField residualVector(output.nCells);
    gamg::multiply(residualVector, ldu, converged);
    const auto residual = maxError(residualVector, source);
    check(residual < 1e-10, "GS residual too large");
    std::cout << "validation," << centreError << ',' << residual << ','
              << solutionError << ',' << crossError << ',' << oneSweepDelta << '\n';
    const auto* experimental = std::getenv("MESH_GS_VARIANT");
    const auto* pmuSelection = std::getenv("MESH_PMU_KERNEL");
    if (!experimental && pmuSelection && (std::string(pmuSelection) == "M7" || std::string(pmuSelection) == "M7.4d" || std::string(pmuSelection) == "M8a" || std::string(pmuSelection) == "M8b" || std::string(pmuSelection) == "M8c" || std::string(pmuSelection) == "M8.4x128" || std::string(pmuSelection) == "M8.2x256" || std::string(pmuSelection) == "M8.1x512" || std::string(pmuSelection) == "SWnd4" || std::string(pmuSelection) == "SWndReload4" || std::string(pmuSelection) == "MinimalReload4" || std::string(pmuSelection) == "Padded4" || std::string(pmuSelection) == "OptimizedMinimalReload4" || std::string(pmuSelection) == "OptimizedMinimalReload4OverlappingMasks" || std::string(pmuSelection) == "OptimizedMinimalReload4SimplifiedMaskIndex" || std::string(pmuSelection) == "OptimizedMinimalReload4RemainingCount" || std::string(pmuSelection) == "OptimizedMinimalReload4SingleIndexGpr" || std::string(pmuSelection) == "OptimizedMinimalReload4SingleIndexGprOverlappingMasks" || std::string(pmuSelection) == "OptimizedMinimalReload4SingleIndexGprMemoryZero" || std::string(pmuSelection) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminal" || std::string(pmuSelection) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminalMemoryZero"))
        experimental = pmuSelection;
    if (!experimental && pmuSelection &&
        (std::string(pmuSelection) == "OptimizedMinimalReload4MaskedScatterBasic" ||
         std::string(pmuSelection) == "OptimizedMinimalReload4MaskedScatterShared"))
        experimental = pmuSelection;
    if (experimental && (std::string(experimental) == "M8a" ||
                         std::string(experimental) == "M8b" ||
                         std::string(experimental) == "M8c" ||
                         std::string(experimental) == "M8.4x128" ||
                         std::string(experimental) == "M8.2x256" ||
                         std::string(experimental) == "M8.1x512" ||
                         std::string(experimental) == "M6.4d-AVX512-A" ||
                         std::string(experimental) == "M6.4d-AVX512-B")) {
        const std::string name(experimental);
        gsExperiment::MinimalOptData data(ldu);
        const bool wide8 = name == "M8.4x128" || name == "M8.2x256" ||
                           name == "M8.1x512";
        const auto validateAndMeasure = [&](auto run) {
            scalarField result(output.nCells,0.0);
            run(result,1);
            check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
                  "M8 one-sweep result differs bitwise");
            run(result,191);
            check(std::memcmp(result.data(),converged.data(),converged.size()*sizeof(double))==0,
                  "M8 192-sweep result differs bitwise");
            std::cout << name << "_validation,bitwise_one_and_192_passed,"
                      << (wide8 ? gsExperiment::m8TerminalRow8(ldu)
                                : data.terminalRow) << '\n';
            if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
            const auto* ctl=std::getenv("MESH_PMU_CONTROL");
            const auto* ackPath=std::getenv("MESH_PMU_ACK");
            if (ctl && ackPath) {
                PerfControl perf(ctl,ackPath);
                measure(name,1,source,run,sweeps,repetitions,loadMs,&perf);
            } else measure(name,1,source,run,sweeps,repetitions,loadMs);
            return 0;
        };
        if (name == "M8.4x128" || name == "M8.2x256" || name == "M8.1x512") {
            check(__builtin_cpu_supports("avx512f") &&
                  __builtin_cpu_supports("avx512vl") &&
                  __builtin_cpu_supports("avx512dq"),
                  "AVX-512F/VL/DQ required");
            if (name == "M8.4x128") {
                const auto run = [&](scalarField& psi,int count) {
                    gsExperiment::minimal8Avx512<0>(psi,ldu,source,count,data);
                };
                return validateAndMeasure(run);
            }
            if (name == "M8.2x256") {
                const auto run = [&](scalarField& psi,int count) {
                    gsExperiment::minimal8Avx512<1>(psi,ldu,source,count,data);
                };
                return validateAndMeasure(run);
            }
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::minimal8Avx512<2>(psi,ldu,source,count,data);
            };
            return validateAndMeasure(run);
        }
        if (name == "M6.4d-AVX512-A" || name == "M6.4d-AVX512-B") {
            check(__builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq"),
                  "AVX-512VL/DQ required");
            auto kernel = name.back()=='A' ? gsExperiment::minimal4Avx512<false>
                                           : gsExperiment::minimal4Avx512<true>;
            return validateAndMeasure([&](scalarField& psi,int count) {
                kernel(psi,ldu,source,count,data);
            });
        }
        if (name == "M8a") {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::cacheHybridM8<0>(psi,ldu,source,count,data);
            };
            return validateAndMeasure(run);
        }
        if (name == "M8b") {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::cacheHybridM8<1>(psi,ldu,source,count,data);
            };
            return validateAndMeasure(run);
        }
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::cacheHybridM8<2>(psi,ldu,source,count,data);
        };
        return validateAndMeasure(run);
    }
    if (experimental &&
        (std::string(experimental) == "OptimizedMinimalReload4MaskedScatterBasic" ||
         std::string(experimental) == "OptimizedMinimalReload4MaskedScatterShared")) {
        const bool shared = std::string(experimental) == "OptimizedMinimalReload4MaskedScatterShared";
        gsExperiment::MaskedScatterData data(ldu);
        const auto run = [&](scalarField& psi, int count) {
            if (shared)
                gsExperiment::optimizedMinimal4MaskedScatter<true>(psi,ldu,source,count,data);
            else
                gsExperiment::optimizedMinimal4MaskedScatter<false>(psi,ldu,source,count,data);
        };
        scalarField result(output.nCells,0.0);
        run(result,1);
        check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
              "masked-scatter one sweep differs bitwise");
        run(result,191);
        check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
              "masked-scatter 192 sweeps differ bitwise");
        std::cout << "masked_scatter_validation," << experimental
                  << ",bitwise_one_and_192_passed," << data.base.terminalRow << '\n';
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
        const auto* ctl=std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath=std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl,ackPath);
            measure(experimental,1,source,run,sweeps,repetitions,loadMs,&perf);
        } else measure(experimental,1,source,run,sweeps,repetitions,loadMs);
        return 0;
    }
    if (experimental && std::string(experimental) == "OptimizedMinimalReload4") {
        gsExperiment::MinimalOptData data(ldu);
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::optimizedMinimal4(psi,ldu,source,count,data);
        };
        scalarField result(output.nCells,0.0);
        run(result,1);
        check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
              "optimized one sweep differs bitwise");
        run(result,191);
        check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
              "optimized 192 sweeps differ bitwise");
        std::cout << "optimized_validation,bitwise_one_and_192_passed," << data.terminalRow << '\n';
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
        const auto* ctl=std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath=std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl,ackPath);
            measure("OptimizedMinimalReload4",1,source,run,sweeps,repetitions,loadMs,&perf);
        } else measure("OptimizedMinimalReload4",1,source,run,sweeps,repetitions,loadMs);
        return 0;
    }
    if (experimental && std::string(experimental) == "OptimizedMinimalReload4OverlappingMasks") {
        gsExperiment::MinimalOptData data(ldu);
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::optimizedMinimal4OverlappingMasks(psi,ldu,source,count,data);
        };
        scalarField result(output.nCells,0.0);
        run(result,1);
        check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
              "overlapping-mask one sweep differs bitwise");
        run(result,191);
        check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
              "overlapping-mask 192 sweeps differ bitwise");
        std::cout << "overlapping_mask_validation,bitwise_one_and_192_passed,"
                  << data.terminalRow << '\n';
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
        const auto* ctl=std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath=std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl,ackPath);
            measure("OptimizedMinimalReload4OverlappingMasks",1,source,run,
                    sweeps,repetitions,loadMs,&perf);
        } else measure("OptimizedMinimalReload4OverlappingMasks",1,source,run,
                       sweeps,repetitions,loadMs);
        return 0;
    }
    if (experimental && std::string(experimental) == "OptimizedMinimalReload4SimplifiedMaskIndex") {
        gsExperiment::MinimalOptData data(ldu);
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::optimizedMinimal4SimplifiedMaskIndex(psi,ldu,source,count,data);
        };
        scalarField result(output.nCells,0.0);
        run(result,1);
        check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
              "simplified-mask-index one sweep differs bitwise");
        run(result,191);
        check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
              "simplified-mask-index 192 sweeps differ bitwise");
        std::cout << "simplified_mask_index_validation,bitwise_one_and_192_passed,"
                  << data.terminalRow << '\n';
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
        const auto* ctl=std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath=std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl,ackPath);
            measure("OptimizedMinimalReload4SimplifiedMaskIndex",1,source,run,
                    sweeps,repetitions,loadMs,&perf);
        } else measure("OptimizedMinimalReload4SimplifiedMaskIndex",1,source,run,
                       sweeps,repetitions,loadMs);
        return 0;
    }
    if (experimental && (std::string(experimental) == "OptimizedMinimalReload4SingleIndexGpr" ||
                         std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprOverlappingMasks" ||
                         std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprMemoryZero" ||
                         std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminal" ||
                         std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminalMemoryZero")) {
        const bool isolatedTerminal =
            std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminal" ||
            std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminalMemoryZero";
        const bool memoryZero =
            std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprMemoryZero" ||
            std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprIsolatedTerminalMemoryZero";
        const bool overlapping = std::string(experimental) ==
            "OptimizedMinimalReload4SingleIndexGprOverlappingMasks" ||
            std::string(experimental) == "OptimizedMinimalReload4SingleIndexGprMemoryZero" ||
            isolatedTerminal;
        gsExperiment::MinimalOptData data(ldu);
        const auto validateAndMeasure = [&](auto run, const char* marker) {
            scalarField result(output.nCells,0.0);
            run(result,1);
            check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
                  "single-index-GPR one sweep differs bitwise");
            run(result,191);
            check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
                  "single-index-GPR 192 sweeps differ bitwise");
            std::cout << marker << ",bitwise_one_and_192_passed,"
                      << data.terminalRow << '\n';
            if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
            const auto* ctl=std::getenv("MESH_PMU_CONTROL");
            const auto* ackPath=std::getenv("MESH_PMU_ACK");
            if (ctl && ackPath) {
                PerfControl perf(ctl,ackPath);
                measure(experimental,1,source,run,sweeps,repetitions,loadMs,&perf);
            } else measure(experimental,1,source,run,sweeps,repetitions,loadMs);
            return 0;
        };
        if (isolatedTerminal && memoryZero) {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::optimizedMinimal4SingleIndexGprIsolatedTerminal<true>(
                    psi,ldu,source,count,data);
            };
            return validateAndMeasure(run,
                "single_index_gpr_isolated_terminal_memory_zero_validation");
        }
        if (isolatedTerminal) {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::optimizedMinimal4SingleIndexGprIsolatedTerminal<false>(
                    psi,ldu,source,count,data);
            };
            return validateAndMeasure(run,
                "single_index_gpr_isolated_terminal_validation");
        }
        if (memoryZero) {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::optimizedMinimal4SingleIndexGprMemoryZero(
                    psi,ldu,source,count,data);
            };
            return validateAndMeasure(run,"single_index_gpr_memory_zero_validation");
        }
        if (overlapping) {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::optimizedMinimal4SingleIndexGpr<true>(
                    psi,ldu,source,count,data);
            };
            return validateAndMeasure(run,
                "single_index_gpr_overlapping_masks_validation");
        }
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::optimizedMinimal4SingleIndexGpr<false>(
                psi,ldu,source,count,data);
        };
        return validateAndMeasure(run,"single_index_gpr_validation");
    }
    if (experimental && std::string(experimental) == "OptimizedMinimalReload4RemainingCount") {
        gsExperiment::MinimalOptData data(ldu);
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::optimizedMinimal4RemainingCount(psi,ldu,source,count,data);
        };
        scalarField result(output.nCells,0.0);
        run(result,1);
        check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
              "remaining-count one sweep differs bitwise");
        run(result,191);
        check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
              "remaining-count 192 sweeps differ bitwise");
        std::cout << "remaining_count_validation,bitwise_one_and_192_passed,"
                  << data.terminalRow << '\n';
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
        const auto* ctl=std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath=std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl,ackPath);
            measure("OptimizedMinimalReload4RemainingCount",1,source,run,
                    sweeps,repetitions,loadMs,&perf);
        } else measure("OptimizedMinimalReload4RemainingCount",1,source,run,
                       sweeps,repetitions,loadMs);
        return 0;
    }
    if (experimental && (std::string(experimental) == "M7" ||
                         std::string(experimental) == "M7.4d")) {
        const bool scalarProductM74d=std::string(experimental)=="M7.4d";
        gsExperiment::MinimalOptData data(ldu);
        const auto validateAndMeasure = [&](auto run, const char* marker) {
            scalarField result(output.nCells,0.0);
            run(result,1);
            check(std::memcmp(result.data(),reference.data(),reference.size()*sizeof(double))==0,
                  "M7 family one-sweep result differs bitwise");
            run(result,191);
            check(std::memcmp(result.data(),converged.data(),reference.size()*sizeof(double))==0,
                  "M7 family 192-sweep result differs bitwise");
            std::cout << marker << ",bitwise_one_and_192_passed,"
                      << data.terminalRow << '\n';
            if (std::getenv("MESH_GS_VALIDATE_ONLY")) return 0;
            const auto* ctl=std::getenv("MESH_PMU_CONTROL");
            const auto* ackPath=std::getenv("MESH_PMU_ACK");
            if (ctl && ackPath) {
                PerfControl perf(ctl,ackPath);
                measure(experimental,1,source,run,sweeps,repetitions,loadMs,&perf);
            } else measure(experimental,1,source,run,sweeps,repetitions,loadMs);
            return 0;
        };
        if (scalarProductM74d) {
            const auto run = [&](scalarField& psi,int count) {
                gsExperiment::scalarProductMinimal4M74d(psi,ldu,source,count,data);
            };
            return validateAndMeasure(run,"m7_4d_validation");
        }
        const auto run = [&](scalarField& psi,int count) {
            gsExperiment::scalarProductMinimal4(psi,ldu,source,count,data);
        };
        return validateAndMeasure(run,"m7_validation");
    }
    if (experimental)
    {
        const std::string name(experimental);
        check(name == "SWndReload4" || name == "SWnd4" || name == "MinimalReload4" || name == "Padded4", "unknown experimental GS variant");
        gsExperiment::Minimal4Data prepared(ldu, name == "Padded4");
        std::cout << "gather_preparation," << name << ',' << prepared.indices.size()
                  << ',' << prepared.preprocessingMs << '\n';
        const auto run = [&](scalarField& psi, int count) {
            if (name == "Padded4") {
                if (psi.size() == source.size()) psi.push_back(0.0);
                gsExperiment::minimal4<true>(psi,ldu,source,count,prepared);
                return;
            }
            if (name == "MinimalReload4") {
                gsExperiment::minimal4<false>(psi,ldu,source,count,prepared);
                return;
            }
            if (name == "SWnd4") gsExperiment::smooth<true>(psi, ldu, source, count, &prepared.bPrime);
            else gsExperiment::smooth<false>(psi, ldu, source, count, &prepared.bPrime);
        };
        scalarField result(output.nCells, 0.0);
        run(result, 1);
        check(std::memcmp(result.data(), reference.data(), reference.size()*sizeof(double)) == 0,
              "experimental GS one sweep differs bitwise");
        run(result, 191);
        check(std::memcmp(result.data(), converged.data(), reference.size()*sizeof(double)) == 0,
              "experimental GS convergence differs bitwise");
        if (name == "Padded4")
            check(result.back() == 0.0 && !std::signbit(result.back()),
                  "Padded4 sentinel changed");
        if (std::getenv("MESH_GS_VALIDATE_ONLY")) {
            std::cout << "experimental_validation," << name
                      << ",bitwise_one_and_192_sweeps_passed\n";
            return 0;
        }
        const auto* ctl = std::getenv("MESH_PMU_CONTROL");
        const auto* ackPath = std::getenv("MESH_PMU_ACK");
        if (ctl && ackPath) {
            PerfControl perf(ctl, ackPath);
            measure(name.c_str(), 1, source, run, sweeps, repetitions, loadMs, &perf);
        } else measure(name.c_str(), 1, source, run, sweeps, repetitions, loadMs);
        return 0;
    }
    const auto* control = std::getenv("MESH_PMU_CONTROL");
    const auto* ack = std::getenv("MESH_PMU_ACK");
    if (control || ack)
    {
        check(control && ack && threads == 1, "perf control requires both FIFOs and single-thread GS");
        const auto* kernel = std::getenv("MESH_PMU_KERNEL");
        const std::string selected = kernel ? kernel : "OriginalGS";
        if (selected == "IH1W1024")
        {
            check(argc >= 9 && std::stoi(argv[8]) == 1024,
                  "IH1 perf control requires width 1024");
            const auto ih1 = fw::Matrix::build(native, 1024);
            const auto packedSource = gather(source, ih1.newToOld());
            const auto packedReference = gather(reference, ih1.newToOld());
            const auto packedConverged = gather(converged, ih1.newToOld());
            const auto smooth = [&](scalarField& psi, int count)
            {
                ih1.smooth(psi, packedSource, count, false);
            };
            scalarField checkPsi(output.nCells, 0.0);
            smooth(checkPsi, 1);
            check(maxError(checkPsi, packedReference) == 0,
                  "IH1 PMU path one-sweep correctness failed");
            smooth(checkPsi, 191);
            check(maxError(checkPsi, packedConverged) < 1e-11,
                  "IH1 PMU path convergence correctness failed");
            PerfControl perf(control, ack);
            measure("IH1W1024", 1, packedSource, smooth, sweeps, repetitions,
                    loadMs, &perf);
            return 0;
        }
        check(selected == "OriginalGS", "unsupported PMU kernel selection");
        PerfControl perf(control, ack);
        measure("OriginalGS", 1, source, gs, sweeps, repetitions, loadMs, &perf);
        return 0;
    }
    for (int arg = 8; arg < argc; ++arg)
    {
        const int width = std::stoi(argv[arg]);
        check(width > 0, "expected positive width");
        const auto start = Clock::now();
        const auto ih1 = fw::Matrix::build(native, width);
        const auto packedSource = gather(source, ih1.newToOld());
        const auto packedReference = gather(reference, ih1.newToOld());
        const auto packedConverged = gather(converged, ih1.newToOld());
        const double setup = millis(start);
        const auto smooth = [&](scalarField& psi, int count)
        {
            ih1.smooth(psi, packedSource, count, false);
        };
        const std::string id = "IH1W" + std::to_string(width);
        const auto verify = [&](const std::string& label, int team, auto solve,
                                const scalarField& shortRef, const scalarField& longRef)
        {
            omp_set_num_threads(team);
            scalarField actual(output.nCells, 0.0);
            solve(actual, 1);
            check(maxError(actual, shortRef) == 0, "IH1 one-sweep exactness failed");
            solve(actual, 191);
            const auto error = maxError(actual, longRef);
            check(error < 1e-11, "IH1 common solution mismatch");
            std::cout << "exact," << label << ',' << team << ',' << error << '\n';
        };
        verify(id, 1, smooth, packedReference, packedConverged);
        if (arg == 8) measure("OriginalGS", 1, source, gs, sweeps, repetitions, loadMs);
        measure(id, 1, packedSource, smooth, sweeps, repetitions, setup);
        if (threads > 1)
        {
            const auto ompStart = Clock::now();
            const Foam::lduMatrix matrix(
                Foam::lduAddressing(native.upperAddr, ldu.ownerStartAddr()),
                native.diag, native.upper, native.upper);
            const auto levels = harness::constructIndexWavefronts(output, matrix, width, false);
            check(harness::validateDependencies(output, levels), "invalid OpenMP dependencies");
            const auto packed = harness::makeWavefrontSchedule(output, matrix, levels);
            const auto blocked = harness::makeBlockedRowDegreeSchedule(packed, 64);
            const double ompSetup = millis(ompStart);
            const std::string ompId = "IndexKahn" + std::to_string(width) + "Blocked64OMP";
            for (int team : {1, threads})
            {
                const auto parallel = [&](scalarField& psi, int count)
                {
                    int detected = 0;
                    wt::persistentOpenMpBlockedRowDegreeSmooth(psi, source, blocked, count, detected);
                    check(detected == team, "OpenMP smoother used a different team size");
                };
                verify(ompId, team, parallel, reference, converged);
                measure(ompId, team, source, parallel, sweeps, repetitions, ompSetup);
            }
        }
    }
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
