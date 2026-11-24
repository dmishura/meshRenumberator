// Sliding read footprints and exact distinct-line stack distances.
#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Word = std::uint64_t;

struct Trace
{
    std::vector<Word> offsets, indices;
    Word bytes;

    std::size_t rows() const { return offsets.size() - 1; }
    Word line(Word address) const { return address / (64 / bytes); }
    Word page(Word address) const { return address / (4096 / bytes); }
};

struct Window
{
    const Trace& trace;
    std::size_t begin = 0, end = 0;
    std::unordered_map<Word, Word> lines, pages;
    std::deque<std::size_t> mins, maxs;

    explicit Window(const Trace& input) : trace(input) {}

    static void remove(std::unordered_map<Word, Word>& counts, Word key)
    {
        const auto it = counts.find(key);
        if (--it->second == 0) counts.erase(it);
    }

    void append()
    {
        const auto address = trace.indices[end];
        ++lines[trace.line(address)];
        ++pages[trace.page(address)];
        while (!mins.empty() && trace.indices[mins.back()] >= address) mins.pop_back();
        while (!maxs.empty() && trace.indices[maxs.back()] <= address) maxs.pop_back();
        mins.push_back(end);
        maxs.push_back(end);
        ++end;
    }

    void advance(std::size_t nextBegin, std::size_t nextEnd)
    {
        for (; begin < std::min(nextBegin, end); ++begin)
        {
            const auto address = trace.indices[begin];
            remove(lines, trace.line(address));
            remove(pages, trace.page(address));
        }
        while (!mins.empty() && mins.front() < nextBegin) mins.pop_front();
        while (!maxs.empty() && maxs.front() < nextBegin) maxs.pop_front();
        begin = nextBegin;
        end = std::max(end, begin);
        while (end < nextEnd) append();
    }

    void write(std::ofstream& out) const
    {
        double result[8]{};
        if (end != begin)
        {
            const auto lo = trace.indices[mins.front()];
            const auto hi = trace.indices[maxs.front()];
            const auto span = hi - lo + 1;
            result[0] = lines.size();
            result[1] = pages.size();
            result[2] = span;
            result[3] = span * trace.bytes;
            result[4] = result[0] / (trace.line(hi) - trace.line(lo) + 1);
            result[5] = end - begin;
            result[6] = end - begin - lines.size();
            result[7] = result[5] / result[0];
        }
        out.write(reinterpret_cast<const char*>(result), sizeof(result));
    }
};

class Fenwick
{
    std::vector<std::int64_t> tree_;

public:
    explicit Fenwick(std::size_t size) : tree_(size + 1) {}

    void add(std::size_t position, int delta)
    {
        for (; position < tree_.size(); position += position & -position)
            tree_[position] += delta;
    }

    Word count(std::size_t position) const
    {
        Word result = 0;
        for (; position; position &= position - 1)
            result += tree_[position];
        return result;
    }
};

std::vector<Word> reuseHistogram(const Trace& trace)
{
    Fenwick active(trace.indices.size());
    std::unordered_map<Word, std::size_t> previous;
    std::vector<Word> histogram(1);
    for (std::size_t position = 1; position <= trace.indices.size(); ++position)
    {
        const auto line = trace.line(trace.indices[position - 1]);
        const auto [it, cold] = previous.try_emplace(line, position);
        if (cold) ++histogram[0];
        else
        {
            // The tree contains exactly the last occurrence of each seen
            // line. Positions beyond this line's timestamp are distinct.
            const auto distance = previous.size() - active.count(it->second);
            if (distance + 1 >= histogram.size()) histogram.resize(distance + 2);
            ++histogram[distance + 1];
            active.add(it->second, -1);
            it->second = position;
        }
        active.add(position, 1);
    }
    return histogram;
}

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

void read(std::ifstream& input, std::vector<Word>& data)
{
    if (!data.empty())
        require(bool(input.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Word))),
                "truncated trace");
}

int main(int argc, char** argv)
try
{
    require(argc >= 8, "usage: backend input output-prefix rows touches entry-bytes reuse K:stride ...");
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "cannot open trace");
    Trace trace{std::vector<Word>(std::stoull(argv[3]) + 1),
                std::vector<Word>(std::stoull(argv[4])), std::stoull(argv[5])};
    require(trace.bytes == 4 || trace.bytes == 8, "unsupported element width");
    read(input, trace.offsets);
    read(input, trace.indices);
    require(input.peek() == EOF && trace.offsets.front() == 0 &&
            trace.offsets.back() == trace.indices.size() &&
            std::is_sorted(trace.offsets.begin(), trace.offsets.end()), "invalid trace");
    require(std::all_of(trace.indices.begin(), trace.indices.end(), [&](Word index) {
        return index < std::numeric_limits<Word>::max() / trace.bytes;
    }), "trace byte span would overflow uint64");
    const std::string prefix(argv[2]);
    for (int arg = 7; arg < argc; ++arg)
    {
        const std::string option(argv[arg]);
        const auto split = option.find(':');
        require(split != std::string::npos, "expected K:stride");
        const auto width = std::stoull(option.substr(0, split));
        const auto stride = std::stoull(option.substr(split + 1));
        require(width && stride, "window and stride must be positive");
        std::ofstream out(prefix + "-" + std::to_string(width), std::ios::binary);
        require(bool(out), "cannot open window output");
        Window window(trace);
        if (width <= trace.rows())
        {
            const auto last = trace.rows() - width;
            for (Word row = 0;; row += stride)
            {
                window.advance(trace.offsets[row], trace.offsets[row + width]);
                window.write(out);
                if (last - row < stride) break;
            }
        }
        require(bool(out), "cannot write window output");
    }
    if (std::string(argv[6]) == "1")
    {
        const auto histogram = reuseHistogram(trace);
        std::ofstream out(prefix + "-reuse", std::ios::binary);
        out.write(reinterpret_cast<const char*>(histogram.data()), histogram.size() * sizeof(Word));
        require(bool(out), "cannot write reuse-distance histogram");
    }
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
