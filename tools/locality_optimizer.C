// Local search over physical cell labels. The standalone helper consumes
// the validated internal endpoints supplied by the Python mesh reader.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Index = std::int32_t;
using Count = std::int64_t;

template<class T>
void unique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct Totals
{
    Count sum = 0;
    long double squares = 0;

    void add(Count value, int sign)
    {
        sum += sign * value;
        squares += sign * static_cast<long double>(value) * value;
    }

    long double objective(int size, long double weight) const
    {
        return (sum / static_cast<long double>(size)) + weight * rms(size);
    }

    long double rms(int size) const
    {
        return std::sqrt(std::max(0.0L, squares) / size);
    }
};

struct Graph
{
    std::vector<Index> starts, neighbours;

    Graph(Index size, const std::vector<Index>& owners, const std::vector<Index>& upper)
        : starts(size + 1), neighbours(2 * owners.size())
    {
        for (std::size_t face = 0; face < owners.size(); ++face)
        {
            const Index a = owners[face], b = upper[face];
            if (a < 0 || a >= b || b >= size) throw std::runtime_error("invalid internal face");
            ++starts[a + 1];
            ++starts[b + 1];
        }
        std::partial_sum(starts.begin(), starts.end(), starts.begin());
        auto next = starts;
        for (std::size_t face = 0; face < owners.size(); ++face)
        {
            const Index a = owners[face], b = upper[face];
            neighbours[next[a]++] = b;
            neighbours[next[b]++] = a;
        }
    }

    Index degree(Index cell) const { return starts[cell+1] - starts[cell]; }
};

// The L0 Original GS kernel reads each owner's contiguous faces twice,
// in the same direction. Each row contributes its second pass plus one
// rewind transition if it spans multiple lines. This tree tracks these for
// every starting alignment modulo 16 faces. Leaf blocks keep memory bounded.
class CoeffTree
{
    static constexpr int blockSize = 32;
    using Phases = std::array<Count, 16>;
    struct Node
    {
        Index length = 0;
        Phases rewind{};
    };
    int base_ = 1;
    std::vector<Node> nodes_;

    void leaf(int block)
    {
        auto& node = nodes_[base_ + block];
        node = {};
        const int begin = block * blockSize;
        const int end = std::min(begin + blockSize, int(rowLengths.size()));
        for (int row = begin; row < end; ++row)
        {
            const Index degree = rowLengths[row];
            if (degree <= 1)
            {
                node.length += degree;
                continue;
            }
            for (int phase = 0; phase < 16; ++phase)
            {
                const Count start = static_cast<Count>(node.length) + phase;
                const Count stop = start + degree - 1;
                const Count upper = stop / 8 - start / 8;
                const Count addressing = stop / 16 - start / 16;
                node.rewind[phase] += upper + (upper > 0) + addressing + (addressing > 0);
            }
            node.length += degree;
        }
    }

    void join(int index)
    {
        const auto& left = nodes_[index * 2];
        const auto& right = nodes_[index * 2 + 1];
        auto& node = nodes_[index];
        node.length = left.length + right.length;
        for (int phase = 0; phase < 16; ++phase)
            node.rewind[phase] = left.rewind[phase] + right.rewind[(phase + left.length) & 15];
    }

public:
    std::vector<Index> rowLengths;

    explicit CoeffTree(std::vector<Index> lengths) : rowLengths(std::move(lengths))
    {
        const int blocks = (rowLengths.size() + blockSize - 1) / blockSize;
        while (base_ < blocks) base_ *= 2;
        nodes_.resize(2 * base_);
        for (int block = 0; block < blocks; ++block) leaf(block);
        for (int node = base_ - 1; node > 0; --node) join(node);
    }

    void refresh(const std::vector<Index>& positions)
    {
        // Positions are sorted. Rebuilding each changed leaf once and
        // propagating its path leaves all other subtrees intact.
        int previous = -1;
        for (const int position : positions)
        {
            const int block = position / blockSize;
            if (block == previous) continue;
            previous = block;
            leaf(block);
            for (int node = (base_ + block) / 2; node > 0; node /= 2) join(node);
        }
    }

    Count score() const
    {
        const Count last = std::max(0, nodes_[1].length - 1);
        return last / 8 + last / 16 + nodes_[1].rewind[0];
    }
};

struct Optimizer
{
    const Graph& graph;
    Index size;
    std::vector<Index> order, labels;
    std::vector<Count> scores;
    std::vector<Index> columns;
    Totals totals;

    explicit Optimizer(const Graph& input)
        : graph(input), size(input.starts.size() - 1), order(size), labels(size), scores(size)
    {
        std::iota(order.begin(), order.end(), 0);
        labels = order;
        for (Index cell = 0; cell < size; ++cell)
        {
            scores[cell] = rowScore(cell);
            totals.add(scores[cell], 1);
        }
    }

    Index outgoing(Index cell) const
    {
        Index count = 0;
        for (Index i = graph.starts[cell]; i < graph.starts[cell+1]; ++i)
            count += labels[graph.neighbours[i]] > labels[cell];
        return count;
    }

    Index adjacentRow(Index position, Index step) const
    {
        do { position += step; }
        while (position >= 0 && position < size && !graph.degree(order[position]));
        return position;
    }

    static Count jump(Index a, Index b)
    {
        const Count distance = std::abs(static_cast<Count>(a) - b);
        return distance > 1 ? distance : 0;
    }

    Count rowScore(Index cell)
    {
        if (!graph.degree(cell)) return 0;
        columns.clear();
        for (Index i = graph.starts[cell]; i < graph.starts[cell+1]; ++i)
            columns.push_back(labels[graph.neighbours[i]]);
        // (owner, neighbour) face sorting makes both incoming and outgoing
        // groups ascending by neighbour label. The groups split at this row.
        std::sort(columns.begin(), columns.end());
        Count result = 0;
        for (std::size_t i = 1; i < columns.size(); ++i)
            result += jump(columns[i-1], columns[i]);
        const Index previous = adjacentRow(labels[cell], -1);
        if (previous >= 0)
        {
            const Index row = order[previous];
            Index last = -1;
            for (Index i = graph.starts[row]; i < graph.starts[row+1]; ++i)
                last = std::max(last, labels[graph.neighbours[i]]);
            result += jump(last, columns.front());
        }
        return result;
    }

    void verify(Totals candidate, Count coeff, const std::vector<Index>* dirty = nullptr)
    {
        Totals full;
        Count start = 0, rewind = 0;
        for (Index position = 0; position < size; ++position)
        {
            const Index cell = order[position];
            const Count value = rowScore(cell);
            if ((!dirty || !std::binary_search(dirty->begin(), dirty->end(), cell)) &&
                scores[cell] != value)
                throw std::runtime_error("affected row set missed a score change");
            full.add(value, 1);
            const Count len = outgoing(cell);
            if (len > 0)
            {
                const Count upper = (start + len - 1) / 8 - start / 8;
                const Count addressing = (start + len - 1) / 16 - start / 16;
                rewind += upper + (upper > 0) + addressing + (addressing > 0);
            }
            start += len;
        }
        const Count last = std::max(Count(0), start - 1);
        if (full.sum != candidate.sum ||
            std::abs(full.squares - candidate.squares) > std::max(0.5L, full.squares * 1e-12L) ||
            coeff != last/8 + last/16 + rewind)
            throw std::runtime_error("incremental objective differs from full recomputation");
    }
};

struct Candidate
{
    std::vector<Index> positions, old, replacement, affected, rows, coeffPositions;
    std::vector<Index> oldLengths;
    std::vector<Count> newScores;

    void prepare(Optimizer& state, Index from, Index to, bool move)
    {
        if (move)
        {
            positions.resize(std::abs(from - to) + 1);
            std::iota(positions.begin(), positions.end(), std::min(from, to));
        }
        else positions = {std::min(from, to), std::max(from, to)};
        old.clear();
        for (const auto position : positions) old.push_back(state.order[position]);
        replacement = old;
        if (move && from < to)
            std::rotate(replacement.begin(), replacement.begin()+1, replacement.end());
        else if (move)
            std::rotate(replacement.begin(), replacement.end()-1, replacement.end());
        else
            std::swap(replacement.front(), replacement.back());

        affected = old;
        for (const auto cell : old)
            affected.insert(affected.end(), state.graph.neighbours.begin() + state.graph.starts[cell],
                            state.graph.neighbours.begin() + state.graph.starts[cell+1]);
        unique(affected);
        rows = affected;
        addSuccessors(state);
    }

    void addSuccessors(const Optimizer& state)
    {
        for (const auto cell : affected)
        {
            const auto following = state.adjacentRow(state.labels[cell], 1);
            if (following < state.size) rows.push_back(state.order[following]);
        }
    }

    void apply(Optimizer& state, const std::vector<Index>& cells) const
    {
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            state.order[positions[i]] = cells[i];
            state.labels[cells[i]] = positions[i];
        }
    }

    Totals evaluate(Optimizer& state)
    {
        addSuccessors(state);
        unique(rows);
        newScores.clear();
        Totals result = state.totals;
        for (const auto cell : rows)
        {
            const Count value = state.rowScore(cell);
            newScores.push_back(value);
            result.add(state.scores[cell], -1);
            result.add(value, 1);
        }
        return result;
    }

    void checkCoefficients(const Optimizer& state, CoeffTree& coeff)
    {
        coeffPositions.clear();
        for (const auto cell : affected) coeffPositions.push_back(state.labels[cell]);
        std::sort(coeffPositions.begin(), coeffPositions.end());
        oldLengths.clear();
        for (const auto position : coeffPositions)
        {
            oldLengths.push_back(coeff.rowLengths[position]);
            coeff.rowLengths[position] = state.outgoing(state.order[position]);
        }
        coeff.refresh(coeffPositions);
    }

    void restoreCoefficients(CoeffTree& coeff) const
    {
        for (std::size_t i = 0; i < coeffPositions.size(); ++i)
            coeff.rowLengths[coeffPositions[i]] = oldLengths[i];
        coeff.refresh(coeffPositions);
    }
};

void metrics(const Optimizer& state, const CoeffTree& coeff, long double weight)
{
    std::cout << "{\"psi_total\":" << state.totals.sum
              << ",\"psi_mean\":" << state.totals.sum / static_cast<long double>(state.size)
              << ",\"psi_rms\":" << state.totals.rms(state.size)
              << ",\"objective\":" << state.totals.objective(state.size, weight)
              << ",\"coefficient_transitions\":" << coeff.score() << '}';
}

int main(int argc, char** argv)
try
{
    if (argc != 12) throw std::runtime_error("input output cells faces candidates window seed patience verify weight tolerance");
    const Index size = std::stoi(argv[3]);
    const auto faceCount = std::stoll(argv[4]);
    const Count budget = std::stoll(argv[5]);
    const int window = std::min(std::stoi(argv[6]), size - 1);
    const auto seed = std::stoull(argv[7]);
    const Count patience = std::stoll(argv[8]);
    const Count verifyEvery = std::stoll(argv[9]);
    const long double weight = std::stold(argv[10]);
    const long double tolerance = std::stold(argv[11]);
    if (size < 1 || faceCount < 0 || faceCount > INT32_MAX / 2 || budget < 0 ||
        window < 0 || patience < 0 || verifyEvery < 0 || !std::isfinite(weight) || weight < 0 ||
        !std::isfinite(tolerance) || tolerance < 0)
        throw std::runtime_error("invalid local search settings");
    const auto start = std::chrono::steady_clock::now();
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<Index> owners(faceCount), neighbours(faceCount);
    const auto bytes = faceCount * sizeof(Index);
    if (!input.read(reinterpret_cast<char*>(owners.data()), bytes) ||
        !input.read(reinterpret_cast<char*>(neighbours.data()), bytes) || input.peek() != EOF)
        throw std::runtime_error("invalid endpoint stream");
    const Graph graph(size, owners, neighbours);
    Optimizer state(graph);
    std::vector<Index> lengths(size);
    for (Index cell = 0; cell < size; ++cell) lengths[cell] = state.outgoing(cell);
    CoeffTree coeff(std::move(lengths));
    state.verify(state.totals, coeff.score());
    const long double limit = coeff.score() * (1 + tolerance);
    std::cout << std::setprecision(18) << "{\"initial\":";
    metrics(state, coeff, weight);
    std::cout << ",\"coefficient_limit\":" << limit << ",\"history\":[";

    Candidate candidate;
    std::mt19937_64 random(seed);
    const auto sample = [&](Index first, Index last) {
        return std::uniform_int_distribution<Index>(first, last)(random);
    };
    const auto near = [&](Index cell) {
        return sample(static_cast<Index>(std::max(Count(0), Count(cell) - window)),
                      static_cast<Index>(std::min(Count(size - 1), Count(cell) + window)));
    };
    Count evaluated = 0, swaps = 0, moves = 0, coeffRejected = 0, lastAccepted = 0;
    bool firstRecord = true;
    const auto record = [&] {
        if (!firstRecord) std::cout << ',';
        firstRecord = false;
        std::cout << "{\"evaluated\":" << evaluated << ",\"accepted\":" << swaps + moves
                  << ",\"metrics\":";
        metrics(state, coeff, weight);
        std::cout << '}';
    };
    record();
    bool stoppedEarly = false;
    while (size > 1 && evaluated < budget)
    {
        const int kind = sample(0, 3);
        const Index from = sample(0, size - 1);
        const Index cell = state.order[from];
        Index to = from;
        if (kind < 2) to = near(from);
        else if (kind == 2 && graph.degree(cell))
        {
            const auto edge = sample(graph.starts[cell], graph.starts[cell+1] - 1);
            to = state.labels[graph.neighbours[edge]];
        }
        else to = sample(0, size - 1);
        ++evaluated;
        if (to != from)
        {
            const bool move = kind == 1;
            candidate.prepare(state, from, to, move);
            candidate.apply(state, candidate.replacement);
            const auto next = candidate.evaluate(state);
            const bool improving = weight == 0 ? next.sum < state.totals.sum :
                next.objective(size, weight) < state.totals.objective(size, weight);
            const bool audit = verifyEvery && evaluated % verifyEvery == 0;
            if (improving || audit)
            {
                candidate.checkCoefficients(state, coeff);
                if (audit) state.verify(next, coeff.score(), &candidate.rows);
            }
            if (improving && coeff.score() <= limit)
            {
                for (std::size_t i = 0; i < candidate.rows.size(); ++i)
                    state.scores[candidate.rows[i]] = candidate.newScores[i];
                state.totals = next;
                if (move) ++moves;
                else ++swaps;
                lastAccepted = evaluated;
            }
            else
            {
                if (improving) ++coeffRejected;
                candidate.apply(state, candidate.old);
                if (improving || audit) candidate.restoreCoefficients(coeff);
            }
        }
        if (evaluated % 10000 == 0) record();
        if (patience && evaluated - lastAccepted >= patience)
        {
            stoppedEarly = true;
            break;
        }
    }
    state.verify(state.totals, coeff.score());
    if (evaluated % 10000 != 0) record();
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(state.order.data()), size * sizeof(Index));
    if (!output) throw std::runtime_error("cannot write permutation");
    std::cout << "],\"final\":";
    metrics(state, coeff, weight);
    std::cout << ",\"evaluated\":" << evaluated << ",\"accepted\":" << swaps + moves
              << ",\"rejected\":" << evaluated - swaps - moves
              << ",\"swaps_accepted\":" << swaps << ",\"moves_accepted\":" << moves
              << ",\"coefficient_rejected\":" << coeffRejected
              << ",\"stopped_early\":" << (stoppedEarly ? "true" : "false")
              << ",\"wall_seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "}\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
