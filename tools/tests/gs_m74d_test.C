#include "GAMGKernels.H"
#include "../gs_minimal_opt.H"
#include "../gs_m74d.H"
#include <cstring>
#include <iostream>
#include <stdexcept>

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
        // Keep the tested row in the ordinary full-load range, with several
        // later rows forcing a terminal suffix at the physical face-array end.
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
        for (int i=0;i<n;++i) rhs[i]=1.0+0.015625*i;
        Foam::scalarField expected(n,0.0), actual(n,0.0);
        smootherTest::gamg::gaussSeidelSmooth(expected,matrix,rhs,1);
        gsExperiment::MinimalOptData data(matrix);
        if (data.terminalRow<=0)
            throw std::runtime_error("synthetic test did not exercise M7.4d main rows");
        gsExperiment::scalarProductMinimal4M74d(actual,matrix,rhs,1,data);
        if (std::memcmp(expected.data(),actual.data(),n*sizeof(double)))
            throw std::runtime_error("M7.4d synthetic one-sweep mismatch");
        smootherTest::gamg::gaussSeidelSmooth(expected,matrix,rhs,191);
        gsExperiment::scalarProductMinimal4M74d(actual,matrix,rhs,191,data);
        if (std::memcmp(expected.data(),actual.data(),n*sizeof(double)))
            throw std::runtime_error("M7.4d synthetic 192-sweep mismatch");
    }
    std::cout << "M7.4d synthetic owner degrees 0..32: bitwise one/192 sweeps passed\n";
}
