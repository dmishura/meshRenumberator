// Global shell-ordered frontier traversal with a bounded recent psi model.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Cell = std::int32_t;
using Count = std::int64_t;

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

struct Graph
{
    std::vector<Cell> start, col;

    Graph(Cell n, const std::vector<Cell>& owner, const std::vector<Cell>& neighbour)
        : start(static_cast<std::size_t>(n) + 1), col(2 * owner.size())
    {
        for (std::size_t i = 0; i < owner.size(); ++i)
        {
            const auto a = owner[i], b = neighbour[i];
            require(a >= 0 && a < b && b < n, "invalid graph endpoint");
            ++start[a + 1];
            ++start[b + 1];
        }
        std::partial_sum(start.begin(), start.end(), start.begin());
        auto next = start;
        for (std::size_t i = 0; i < owner.size(); ++i)
        {
            const auto a = owner[i], b = neighbour[i];
            col[next[a]++] = b;
            col[next[b]++] = a;
        }
    }

    auto begin(Cell v) const { return col.begin() + start[v]; }
    auto end(Cell v) const { return col.begin() + start[v+1]; }
    Cell count(Cell v) const { return start[v+1] - start[v]; }
    Cell size() const { return static_cast<Cell>(start.size() - 1); }
};

template<class T>
void dedup(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct Recent
{
    const Graph& graph;
    const std::vector<Cell>& label;
    const std::vector<Cell>& order;
    Cell rows;
    std::vector<Count> lines, pending;
    std::size_t distinctLines = 0, distinctPending = 0;

    Recent(const Graph& g, const std::vector<Cell>& labels, const std::vector<Cell>& ord, Cell r)
        : graph(g), label(labels), order(ord), rows(r),
          lines((static_cast<std::size_t>(g.size()) + 7) / 8), pending(g.size()) {}

    static void update(std::vector<Count>& counters, std::size_t& distinct, Cell slot, Count delta)
    {
        auto& value = counters[slot];
        if (!value && delta > 0) ++distinct;
        value += delta;
        require(value >= 0, "recent read count underflow");
        if (!value && delta < 0) --distinct;
    }

    void place(Cell v)
    {
        const Cell at = label[v];
        // Previous rows that read v now resolve its line. Count the write
        // to v as a touch to that same line.
        update(lines, distinctLines, at / 8, 1 + pending[v]);
        update(pending, distinctPending, v, -pending[v]);
        for (auto edge = graph.begin(v); edge != graph.end(v); ++edge)
            if (label[*edge] < 0) update(pending, distinctPending, *edge, 1);

        if (at >= rows)
        {
            const Cell old = order[at - rows];
            update(lines, distinctLines, label[old] / 8, -1);
            for (auto edge = graph.begin(old); edge != graph.end(old); ++edge)
            {
                const auto other = label[*edge];
                if (other < 0) update(pending, distinctPending, *edge, -1);
                else if (other > label[old]) update(lines, distinctLines, other / 8, -1);
            }
        }
    }

    void audit() const
    {
        std::vector<Count> expectedLines(lines.size()), expectedPending(pending.size());
        const std::size_t begin = order.size() > static_cast<std::size_t>(rows)
            ? order.size() - rows : 0;
        for (std::size_t at = begin; at < order.size(); ++at)
        {
            const auto v = order[at];
            ++expectedLines[at / 8];
            for (auto i = graph.begin(v); i != graph.end(v); ++i)
            {
                const auto other = label[*i];
                if (other < 0) ++expectedPending[*i];
                else if (static_cast<std::size_t>(other) > at) ++expectedLines[other / 8];
            }
        }
        require(expectedLines == lines && expectedPending == pending,
                "incremental recent-row cache model differs from full recomputation");
        const auto occupied = [](const auto& counts) {
            return static_cast<std::size_t>(std::count_if(counts.begin(), counts.end(),
                                                         [](Count v) { return v > 0; }));
        };
        require(occupied(expectedLines) == distinctLines &&
                occupied(expectedPending) == distinctPending, "incremental footprint size mismatch");
    }
};

struct RootSearch
{
    const Graph& graph;
    std::vector<Cell> distance, parent, queue;
    explicit RootSearch(const Graph& g) : graph(g), distance(g.size(), -1), parent(g.size(), -1) {}

    Cell farthest(Cell start)
    {
        for (auto v : queue) distance[v] = -1;
        queue.clear();
        queue.push_back(start);
        distance[start] = 0;
        parent[start] = -1;
        Cell best = start;
        for (std::size_t i = 0; i < queue.size(); ++i)
        {
            const auto v = queue[i];
            if (distance[v] > distance[best] || (distance[v] == distance[best] && v < best))
                best = v;
            for (auto edge = graph.begin(v); edge != graph.end(v); ++edge)
            {
                const auto u = *edge;
                if (distance[u] >= 0) continue;
                distance[u] = distance[v] + 1;
                parent[u] = v;
                queue.push_back(u);
            }
        }
        return best;
    }

    Cell find(Cell seed, const std::string& rule)
    {
        if (rule == "seed") return seed;
        auto far = farthest(seed);
        if (rule == "peripheral") return far;
        far = farthest(far);
        for (Cell steps = distance[far] / 2; steps > 0; --steps) far = parent[far];
        return far;
    }
};

struct Traversal
{
    const Graph& graph;
    std::vector<Cell> order, label, shell, shellDepth;
    std::deque<Cell> frontier;
    std::vector<Cell> assignedLines, future;
    std::vector<Cell> roots;
    std::array<double, 4> termSum{};
    std::size_t maxFrontier = 0, maxRecentLines = 0, maxPending = 0;
    Count scored = 0, audited = 0;

    explicit Traversal(const Graph& g)
        : graph(g), label(g.size(), -1), shell(g.size(), -1), shellDepth(g.size())
    {
        order.reserve(g.size());
    }

    struct Score
    {
        std::array<double, 4> terms;
        Cell chain;
    };

    Score evaluate(Cell v, std::size_t rank, Cell scan, const Recent& recent)
    {
        const auto at = static_cast<Cell>(order.size());
        assignedLines.clear();
        future.clear();
        assignedLines.push_back(at / 8);
        Cell earliest = at, chain = 0;
        Count assigned = 0;
        for (auto edge = graph.begin(v); edge != graph.end(v); ++edge)
        {
            const auto u = *edge;
            if (label[u] >= 0)
            {
                assignedLines.push_back(label[u] / 8);
                earliest = std::min(earliest, label[u]);
                ++assigned;
                if (shell[u] == shell[v]) chain = std::max(chain, shellDepth[u]);
            }
            else future.push_back(u);
        }
        dedup(assignedLines);
        dedup(future);

        const auto reusedLines = std::count_if(assignedLines.begin(), assignedLines.end(),
                                               [&](Cell l) { return recent.lines[l] > 0; });
        const auto reusedFuture = std::count_if(future.begin(), future.end(),
                                                [&](Cell u) { return recent.pending[u] > 0; });
        // One eighth of a line per future cell is a dense graph proxy.
        // Original cell labels are never converted to candidate addresses.
        const double scale = assignedLines.size() + future.size() / 8.0;
        const double reused = reusedLines + reusedFuture / 8.0;
        const double span = std::min(1.0, (at - earliest) / static_cast<double>(recent.rows));
        const double exposure = (future.size() - reusedFuture) /
                                static_cast<double>(std::max<Cell>(1, graph.count(v)));
        const double contact = graph.count(v) ? assigned / static_cast<double>(graph.count(v)) : 0;
        const double compact = 0.55 * std::min(1.0, chain / 8.0)
                             + 0.25 * (rank / static_cast<double>(std::max<Cell>(1, scan - 1)))
                             + 0.15 * exposure + 0.05 * (1.0 - contact);
        return {{std::min(1.0, (scale - reused) / 8.0), std::min(1.0, reused / 8.0),
                 span, compact}, chain + 1};
    }

    void run(Cell scan, Cell recentRows, std::uint64_t seed, const std::string& rootRule,
             const std::array<double, 4>& weights, Count auditEvery)
    {
        Recent recent(graph, label, order, recentRows);
        RootSearch rootSearch(graph);
        Cell next = 0;
        while (order.size() < label.size())
        {
            if (frontier.empty())
            {
                Cell initial;
                if (order.empty()) initial = seed % label.size();
                else
                {
                    while (next < graph.size() && label[next] >= 0) ++next;
                    require(next < graph.size(), "no unassigned restart cell");
                    initial = next;
                }
                const auto root = rootSearch.find(initial, rootRule);
                require(label[root] < 0, "restart reached an already assigned component");
                roots.push_back(root);
                shell[root] = 0;
                frontier.push_back(root);
            }
            maxFrontier = std::max(maxFrontier, frontier.size());
            const auto activeShell = shell[frontier.front()];
            std::size_t best = 0;
            Cell bestCell = frontier.front();
            double bestValue = std::numeric_limits<double>::infinity();
            Score bestScore{};
            for (std::size_t j = 0; j < frontier.size() && j < static_cast<std::size_t>(scan)
                 && shell[frontier[j]] == activeShell; ++j)
            {
                const auto current = frontier[j];
                const auto score = evaluate(current, j, scan, recent);
                const auto& p = score.terms;
                const auto value = weights[0]*p[0] - weights[1]*p[1] +
                                   weights[2]*p[2] + weights[3]*p[3];
                ++scored;
                if (value < bestValue || (value == bestValue && current < bestCell))
                {
                    best = j;
                    bestCell = current;
                    bestValue = value;
                    bestScore = score;
                }
            }
            frontier.erase(frontier.begin() + best);
            require(label[bestCell] < 0, "frontier cell already assigned");
            label[bestCell] = order.size();
            shellDepth[bestCell] = bestScore.chain;
            order.push_back(bestCell);
            for (std::size_t i = 0; i < termSum.size(); ++i) termSum[i] += bestScore.terms[i];
            recent.place(bestCell);
            maxRecentLines = std::max(maxRecentLines, recent.distinctLines);
            maxPending = std::max(maxPending, recent.distinctPending);
            if (auditEvery && order.size() % auditEvery == 0)
            {
                recent.audit();
                ++audited;
            }
            for (auto edge = graph.begin(bestCell); edge != graph.end(bestCell); ++edge)
            {
                if (shell[*edge] >= 0) continue;
                shell[*edge] = activeShell + 1;
                frontier.push_back(*edge);
            }
        }
    }
};

int main(int argc, char** argv)
try
{
    require(argc == 14, "usage: backend endpoints output cells faces scan recent seed root audit wNew wReuse wSpan wFront");
    const Cell n = std::stoi(argv[3]);
    const auto m = std::stoll(argv[4]);
    const Cell scan = std::stoi(argv[5]);
    const Cell recent = std::stoi(argv[6]);
    const auto seed = std::stoull(argv[7]);
    const std::string root(argv[8]);
    const auto audit = std::stoll(argv[9]);
    std::array<double, 4> weights{};
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = std::stod(argv[10+i]);
        require(std::isfinite(weights[i]) && weights[i] >= 0 && weights[i] <= 1000, "invalid score weight");
    }
    require(n > 0 && n < std::numeric_limits<Cell>::max() && m >= 0 &&
            m <= std::numeric_limits<Cell>::max() / 2 && scan > 0 && recent > 0 &&
            audit >= 0 && (root == "seed" || root == "peripheral" || root == "centre"),
            "invalid graph or traversal settings");
    std::vector<Cell> owner(m), neighbour(m);
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "cannot open graph input");
    const auto bytes = m * sizeof(Cell);
    if (m)
        require(bool(input.read(reinterpret_cast<char*>(owner.data()), bytes)) &&
                bool(input.read(reinterpret_cast<char*>(neighbour.data()), bytes)), "truncated graph");
    require(input.peek() == EOF, "unexpected graph input");
    const Graph graph(n, owner, neighbour);
    Traversal traversal(graph);
    traversal.run(scan, recent, seed, root, weights, audit);
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(traversal.order.data()), static_cast<Count>(n) * sizeof(Cell));
    require(bool(out), "cannot save permutation");
    std::cout << std::setprecision(12) << "{\"disconnected_restarts\":" << traversal.roots.size() - 1
              << ",\"first_root\":" << traversal.roots.front()
              << ",\"max_frontier\":" << traversal.maxFrontier
              << ",\"max_recent_known_lines\":" << traversal.maxRecentLines
              << ",\"max_recent_pending_cells\":" << traversal.maxPending
              << ",\"scored_candidates\":" << traversal.scored
              << ",\"audited_steps\":" << traversal.audited << ",\"selected_term_mean\":[";
    for (std::size_t i = 0; i < traversal.termSum.size(); ++i)
    {
        if (i) std::cout << ',';
        std::cout << traversal.termSum[i] / n;
    }
    std::cout << "]}\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
