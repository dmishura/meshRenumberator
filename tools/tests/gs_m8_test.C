#include "GAMGKernels.H"
#include "../gs_minimal_opt.H"
#include "../gs_m7.H"
#include "../gs_m74d.H"
#include "../gs_m8.H"
#include <cstring>
#include <iostream>
#include <stdexcept>

template<int Variant>
void checkVariant(const smootherTest::gamg::GAMGMatrix& matrix,
                  const Foam::scalarField& rhs)
{
    const int n=matrix.nCells();
    Foam::scalarField expected(n,0.0), actual(n,0.0);
    smootherTest::gamg::gaussSeidelSmooth(expected,matrix,rhs,1);
    gsExperiment::MinimalOptData data(matrix);
    gsExperiment::cacheHybridM8<Variant>(actual,matrix,rhs,1,data);
    if (std::memcmp(expected.data(),actual.data(),n*sizeof(double)))
        throw std::runtime_error("M8 one-sweep mismatch");
    smootherTest::gamg::gaussSeidelSmooth(expected,matrix,rhs,191);
    gsExperiment::cacheHybridM8<Variant>(actual,matrix,rhs,191,data);
    if (std::memcmp(expected.data(),actual.data(),n*sizeof(double)))
        throw std::runtime_error("M8 192-sweep mismatch");
}

int main()
{
    for (int degree=0; degree<=32; ++degree) {
        constexpr int n=40;
        Foam::labelField owners, neighbours;
        Foam::scalarField coefficients;
        for (int j=0; j<degree; ++j) {
            owners.push_back(0);
            neighbours.push_back(j+1);
            coefficients.push_back((j%5==0) ? 0.0 : (j%2 ? -0.125 : 0.0625));
        }
        // Include full and tail chunks near the physical face-array end.
        for (int owner=33; owner<=35; ++owner) {
            for (int j=1; j<=4; ++j) {
                owners.push_back(owner);
                neighbours.push_back(owner+j);
                coefficients.push_back((j%2) ? -0.03125 : 0.0);
            }
        }
        smootherTest::gamg::GAMGMatrix matrix(
            n,owners,neighbours,Foam::scalarField(n,4.0),coefficients);
        Foam::scalarField rhs(n);
        for (int i=0; i<n; ++i) rhs[i]=1.0+0.015625*i;
        if (gsExperiment::MinimalOptData(matrix).terminalRow<=0)
            throw std::runtime_error("synthetic case does not exercise ordinary rows");
        checkVariant<0>(matrix,rhs);
        checkVariant<1>(matrix,rhs);
        checkVariant<2>(matrix,rhs);
    }
    std::cout << "M8a/b/c synthetic owner degrees 0..32: bitwise one/192 sweeps passed\n";
}
