// Adapter for the existing GAMG L0 GS and IH1 kernels.
#include "GAMGKernels.H"
#include "ScheduleBuilder.H"
#include "fwMatrix.H"

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
    const auto* control = std::getenv("MESH_PMU_CONTROL");
    const auto* ack = std::getenv("MESH_PMU_ACK");
    if (control || ack)
    {
        check(control && ack && threads == 1, "perf control requires both FIFOs and single-thread GS");
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
