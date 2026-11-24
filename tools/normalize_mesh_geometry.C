// Normalize geometry with the same parser used by the established benchmark.
#include "PolyMeshReader.C"
#include <iostream>

namespace smootherTest
{
template<class T>
void writeGeometryArray(std::ofstream& out, const std::vector<T>& data)
{
    out << data.size() << "\n(";
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    out << ")\n";
}

void convertGeometry(const std::filesystem::path& source, const std::filesystem::path& output)
{
    static_assert(sizeof(int) == 4 && sizeof(double) == 8);
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error This converter writes little endian payloads
#endif
    {
        const auto [starts, vertices] = readCompactFaces(source/"faces");
        std::ofstream out(output/"faces", std::ios::binary);
        out << "FoamFile\n{ version 2.0; format binary; arch \"LSB;label=32;scalar=64\";"
            << " class faceCompactList; location \"constant/polyMesh\"; object faces; }\n";
        writeGeometryArray(out, starts);
        writeGeometryArray(out, vertices);
        if (!out) throw std::runtime_error("cannot write faces");
    }
    {
        const auto points = readPoints(source/"points");
        static_assert(sizeof(points.front()) == 3 * sizeof(double));
        std::ofstream out(output/"points", std::ios::binary);
        out << "FoamFile\n{ version 2.0; format binary; arch \"LSB;label=32;scalar=64\";"
            << " class vectorField; location \"constant/polyMesh\"; object points; }\n";
        writeGeometryArray(out, points);
        if (!out) throw std::runtime_error("cannot write points");
    }
}
}

int main(int argc, char** argv)
try
{
    if (argc != 3) throw std::runtime_error("usage: normalize_geometry inputPolyMesh outputPolyMesh");
    smootherTest::convertGeometry(argv[1], argv[2]);
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
