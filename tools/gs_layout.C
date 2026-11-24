// Inspect the logical memory streams of the production IH1 packed matrix.
#include "fwMatrix.H"
#include "ScheduleBuilder.H"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>

void scheduleStatistics(std::vector<std::size_t> widths, std::size_t entries,
                        std::size_t blocks = 0)
{
    if (widths.empty()) throw std::runtime_error("empty schedule");
    std::sort(widths.begin(), widths.end());
    const std::size_t n = widths.size(), middle = n / 2;
    const double median = n % 2 ? widths[middle] : (widths[middle-1] + widths[middle]) / 2.0;
    const auto smallEnd = std::lower_bound(widths.begin(), widths.end(), 64);
    const auto smallRows = std::accumulate(widths.begin(), smallEnd, std::size_t{});
    const auto rows = std::accumulate(widths.begin(), widths.end(), std::size_t{});
    std::cout << "{\"levels\":" << n << ",\"mean_rows\":" << double(rows) / n
              << ",\"minimum_rows\":" << widths.front() << ",\"median_rows\":" << median
              << ",\"p90_rows\":" << widths[(n - 1) * 9 / 10]
              << ",\"maximum_rows\":" << widths.back()
              << ",\"levels_below_64_rows\":" << smallEnd - widths.begin()
              << ",\"fraction_rows_in_waves_below_64\":" << double(smallRows) / rows
              << ",\"logical_wave_boundaries\":" << n - 1
              << ",\"work\":{\"row_updates\":" << rows
              << ",\"coefficient_entries\":" << entries << ",\"blocks\":" << blocks << "}}";
}

int main(int argc, char** argv)
try
{
    if (argc != 3) throw std::runtime_error("usage: gs_layout polyMesh width");
    const auto width = std::stoi(argv[2]);
    if (width <= 0) throw std::runtime_error("invalid schedule width");
    const auto mesh = smootherTest::PolyMeshReader::read(argv[1]);
    const auto ldu = smootherTest::harness::makeMatrix(mesh);
    const auto packed = smootherTest::fwgamg::Matrix::build(
        smootherTest::fwgamg::nativeMatrixData(ldu), width);
    const auto& columns = packed.cols();
    if (columns.size() != packed.coeffs().size() ||
        packed.rowStarts().size() != static_cast<std::size_t>(packed.nCells()) + 1 ||
        packed.rowStarts().back() != static_cast<Foam::label>(columns.size()))
        throw std::runtime_error("unexpected packed layout");

    std::int64_t jumps = 0, nonsequential = 0, lines = 0, pages = 0;
    for (std::size_t i = 1; i < columns.size(); ++i)
    {
        const auto previous = columns[i-1];
        const auto next = columns[i];
        const auto delta = std::abs(static_cast<std::int64_t>(next) - previous);
        jumps += delta > 1 ? delta : 0;
        nonsequential += delta > 1;
        lines += previous / 8 != next / 8;
        pages += previous / 512 != next / 512;
    }
    std::int64_t rwLines = 0, rwPages = 0;
    Foam::label previous = -1;
    const auto touch = [&](Foam::label cell)
    {
        if (previous >= 0)
        {
            rwLines += previous / 8 != cell / 8;
            rwPages += previous / 512 != cell / 512;
        }
        previous = cell;
    };
    for (Foam::label row = 0; row < packed.nCells(); ++row)
    {
        for (auto pos = packed.rowStarts()[row]; pos < packed.rowStarts()[row + 1]; ++pos)
            touch(columns[pos]);
        touch(row);
    }
    const auto index = smootherTest::harness::constructIndexWavefronts(mesh, ldu, width, false);
    const auto aligned = smootherTest::harness::makeWavefrontSchedule(mesh, ldu, index);
    const auto blocked = smootherTest::harness::makeBlockedRowDegreeSchedule(aligned, 64);
    const auto minimal = smootherTest::harness::constructMinimalWavefronts(mesh);
    std::vector<std::size_t> packedWidths;
    for (std::size_t i = 1; i < packed.levelStarts().size(); ++i)
        packedWidths.push_back(packed.levelStarts()[i] - packed.levelStarts()[i - 1]);
    const auto last = std::max<std::int64_t>(0, static_cast<std::int64_t>(columns.size()) - 1);
    const auto lastRow = packed.nCells() - 1;
    std::cout << "{\"width\":" << width
              << ",\"line_bytes\":64,\"page_bytes\":4096"
              << ",\"critical_path_rows\":" << minimal.widths.size()
              << ",\"index_kahn_wavefronts\":";
    scheduleStatistics(index.widths, aligned.coeffs.size(), blocked.blockRowStarts.size());
    std::cout << ",\"packed_ih1_wavefronts\":";
    scheduleStatistics(std::move(packedWidths), packed.coeffs().size());
    std::cout << ",\"packed_ih1_levels\":" << packed.levelStarts().size() - 1
              << ",\"psi\":{\"reads\":" << columns.size()
              << ",\"total_nonsequential_jump\":" << jumps
              << ",\"nonsequential_transitions\":" << nonsequential
              << ",\"cache_line_transitions\":" << lines
              << ",\"page_transitions\":" << pages << "}"
              << ",\"psi_read_write\":{\"reads\":" << columns.size()
              << ",\"writes\":" << packed.nCells()
              << ",\"cache_line_transitions\":" << rwLines
              << ",\"page_transitions\":" << rwPages << "}"
              << ",\"coeffs\":{\"entries\":" << columns.size()
              << ",\"cache_line_transitions\":" << last / 8
              << ",\"page_transitions\":" << last / 512 << "}"
              << ",\"cols\":{\"entries\":" << columns.size()
              << ",\"cache_line_transitions\":" << last / 16
              << ",\"page_transitions\":" << last / 1024 << "}"
              << ",\"diag\":{\"entries\":" << packed.nCells()
              << ",\"cache_line_transitions\":" << lastRow / 8
              << ",\"page_transitions\":" << lastRow / 512 << "}"
              << ",\"rowOffsets\":{\"entries\":" << packed.rowStarts().size()
              << ",\"cache_line_transitions\":" << packed.nCells() / 16
              << ",\"page_transitions\":" << packed.nCells() / 1024 << "}}\n";
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
