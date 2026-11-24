// Connected graph growth and ordering inside each partition.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

using Label = std::int32_t;
using Count = std::int64_t;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Graph
{
    std::vector<Label> starts, cols;

    Graph(Label n, const std::vector<Label>& owners, const std::vector<Label>& neighbours)
        : starts(static_cast<std::size_t>(n) + 1), cols(2 * owners.size())
    {
        for (std::size_t i = 0; i < owners.size(); ++i)
        {
            const auto a = owners[i], b = neighbours[i];
            require(a >= 0 && a < b && b < n, "invalid adjacency endpoint");
            ++starts[a + 1];
            ++starts[b + 1];
        }
        std::partial_sum(starts.begin(), starts.end(), starts.begin());
        auto next = starts;
        for (std::size_t i = 0; i < owners.size(); ++i)
        {
            const auto a = owners[i], b = neighbours[i];
            cols[next[a]++] = b;
            cols[next[b]++] = a;
        }
    }

    Label degree(Label cell) const { return starts[cell + 1] - starts[cell]; }
};

std::uint64_t mix(std::uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Choice
{
    Count priority;
    std::uint64_t tie;
    Label cell;

    bool operator<(const Choice& other) const
    {
        return std::tie(priority, tie, cell) < std::tie(other.priority, other.tie, other.cell);
    }
};

struct Partitions
{
    std::vector<Label> membership, traversal, localRcm, offsets{0}, footprintCells;
    std::vector<Count> estimatedBytes;

    Partitions(const Graph& graph, Count budget, std::uint64_t seed)
        : membership(graph.starts.size() - 1, -1)
    {
        const auto n = static_cast<Label>(membership.size());
        traversal.reserve(n);
        std::vector<Label> footprint(n, -1);
        std::vector<Label> examined(n, -1);
        for (Label root = 0; root < n; ++root)
        {
            if (membership[root] >= 0) continue;
            const Label group = footprintCells.size();
            std::priority_queue<Choice> frontier;
            Count estimated = 4; // terminating row offset
            Label footprintSize = 0;
            const auto properties = [&](Label cell)
            {
                Label inside = 0, added = footprint[cell] != group;
                for (Label i = graph.starts[cell]; i < graph.starts[cell + 1]; ++i)
                {
                    const auto other = graph.cols[i];
                    inside += membership[other] == group;
                    added += footprint[other] != group;
                }
                return std::pair<Count, Count>(inside, added);
            };
            const auto choice = [&](Label cell, Count inside, Count added)
            {
                // Reward internal contacts and read overlap. Penalize the
                // net boundary growth and the increase in the read set.
                const auto boundary = graph.degree(cell) - 2 * inside;
                const auto overlap = graph.degree(cell) + 1 - added;
                return Choice{8 * inside + overlap - 2 * boundary - 2 * added,
                              mix(seed ^ static_cast<std::uint64_t>(cell)), cell};
            };
            const auto add = [&](Label cell)
            {
                membership[cell] = group;
                traversal.push_back(cell);
                estimated += 4 + 4 * static_cast<Count>(graph.degree(cell));
                const auto mark = [&](Label other)
                {
                    if (footprint[other] != group)
                    {
                        footprint[other] = group;
                        ++footprintSize;
                        estimated += 8;
                    }
                };
                mark(cell);
                for (Label i = graph.starts[cell]; i < graph.starts[cell + 1]; ++i)
                    mark(graph.cols[i]);

                // One frontier entry per newly discovered cell. Its
                // priority is refreshed lazily as the read set changes.
                for (Label i = graph.starts[cell]; i < graph.starts[cell + 1]; ++i)
                {
                    const auto other = graph.cols[i];
                    if (membership[other] >= 0 || examined[other] == group) continue;
                    examined[other] = group;
                    const auto [inside, added] = properties(other);
                    frontier.push(choice(other, inside, added));
                }
            };

            add(root);
            while (!frontier.empty() && estimated < budget)
            {
                const auto candidate = frontier.top();
                frontier.pop();
                const auto [inside, added] = properties(candidate.cell);
                // Parallel faces to one neighbour may overcount potential
                // growth. Deduplicate on a budget decision only.
                Count cost = 4 + 4 * static_cast<Count>(graph.degree(candidate.cell)) + 8 * added;
                if (cost > budget - estimated)
                {
                    std::vector<Label> missing;
                    if (footprint[candidate.cell] != group) missing.push_back(candidate.cell);
                    for (Label i = graph.starts[candidate.cell]; i < graph.starts[candidate.cell + 1]; ++i)
                        if (footprint[graph.cols[i]] != group) missing.push_back(graph.cols[i]);
                    std::sort(missing.begin(), missing.end());
                    const auto distinct = std::unique(missing.begin(), missing.end()) - missing.begin();
                    cost = 4 + 4 * static_cast<Count>(graph.degree(candidate.cell)) + 8 * distinct;
                    if (cost > budget - estimated) continue;
                }
                const auto updated = choice(candidate.cell, inside, added);
                if (updated.priority != candidate.priority)
                    frontier.push(updated);
                else
                    add(candidate.cell);
            }
            offsets.push_back(traversal.size());
            footprintCells.push_back(footprintSize);
            estimatedBytes.push_back(estimated);
        }
        require(traversal.size() == membership.size(), "incomplete partition coverage");
        orderLocal(graph);
    }

    void orderLocal(const Graph& graph)
    {
        std::vector<Label> degrees(membership.size(), 0);
        std::vector<bool> visited(membership.size());
        for (std::size_t cell = 0; cell < membership.size(); ++cell)
            for (Label i = graph.starts[cell]; i < graph.starts[cell + 1]; ++i)
                degrees[cell] += membership[graph.cols[i]] == membership[cell];
        localRcm.reserve(membership.size());
        std::vector<Label> neighbours;
        const auto byDegree = [&](Label a, Label b)
        {
            return std::tie(degrees[a], a) < std::tie(degrees[b], b);
        };
        for (std::size_t group = 0; group < footprintCells.size(); ++group)
        {
            const auto begin = offsets[group], end = offsets[group + 1];
            const auto root = *std::min_element(traversal.begin() + begin, traversal.begin() + end,
                                                byDegree);
            localRcm.push_back(root);
            visited[root] = true;
            for (auto cursor = begin; cursor < static_cast<Label>(localRcm.size()); ++cursor)
            {
                const auto cell = localRcm[cursor];
                neighbours.clear();
                for (Label i = graph.starts[cell]; i < graph.starts[cell + 1]; ++i)
                {
                    const auto other = graph.cols[i];
                    if (!visited[other] && membership[other] == static_cast<Label>(group))
                    {
                        neighbours.push_back(other);
                        visited[other] = true;
                    }
                }
                std::sort(neighbours.begin(), neighbours.end(), byDegree);
                localRcm.insert(localRcm.end(), neighbours.begin(), neighbours.end());
            }
            require(static_cast<Label>(localRcm.size()) == end, "partition is disconnected");
            std::reverse(localRcm.begin() + begin, localRcm.begin() + end);
        }
    }
};

template<class T>
void write(std::ofstream& stream, const std::vector<T>& data)
{
    stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

int main(int argc, char** argv)
try
{
    require(argc == 7, "usage: partition-backend input output cells faces budget-bytes seed");
    const auto n = std::stoi(argv[3]);
    const auto f = std::stoll(argv[4]);
    const auto budget = std::stoll(argv[5]);
    const auto seed = std::stoull(argv[6]);
    require(n > 0 && n < std::numeric_limits<Label>::max() && f >= 0 &&
            f <= std::numeric_limits<Label>::max() / 2 && budget > 0, "invalid graph size or budget");
    std::vector<Label> owner(f), neighbour(f);
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "cannot open graph");
    if (f)
    {
        const auto bytes = f * sizeof(Label);
        require(bool(input.read(reinterpret_cast<char*>(owner.data()), bytes)) &&
                bool(input.read(reinterpret_cast<char*>(neighbour.data()), bytes)), "truncated graph");
    }
    require(input.peek() == EOF, "unexpected trailing graph data");
    const Graph graph(n, owner, neighbour);
    const Partitions result(graph, budget, seed);
    std::ofstream output(argv[2], std::ios::binary);
    write(output, std::vector<Label>{static_cast<Label>(result.footprintCells.size())});
    write(output, result.membership);
    write(output, result.traversal);
    write(output, result.localRcm);
    write(output, result.offsets);
    write(output, result.footprintCells);
    write(output, result.estimatedBytes);
    require(bool(output), "cannot write partition results");
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
