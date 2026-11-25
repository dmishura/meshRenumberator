#include "GAMGKernels.H"
#include "../gs_window4.H"
#include "../gs_minimal4.H"
#include "../gs_minimal_opt.H"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <cfenv>
#include <cmath>
__attribute__((target("avx2,no-fma")))
void inactiveLaneTest() {
    double psi[2]={std::numeric_limits<double>::infinity(),2.0};
    std::feclearexcept(FE_ALL_EXCEPT);
    const double previous=-0.0;
    const double value=gsExperiment::lane(previous,0.0,0,0,psi);
    if (std::memcmp(&previous,&value,8) || std::fetestexcept(FE_INVALID))
        throw std::runtime_error("inactive infinity or signed zero mismatch");
    psi[0]=std::numeric_limits<double>::quiet_NaN();
    if (gsExperiment::lane(1.0,0.0,0,0,psi)!=1.0)
        throw std::runtime_error("inactive NaN propagated");
}
int main() {
    inactiveLaneTest();
    {
        // Distinct source coefficients expose any scatter access to the
        // original matrix instead of the shared prepared representation.
        const Foam::labelField owners{0,0,1}, neighbours{1,2,3};
        const Foam::scalarField diagonal(4,4.0);
        smootherTest::gamg::GAMGMatrix original(4,owners,neighbours,diagonal,
                                               Foam::scalarField{0.1,0.2,0.3});
        smootherTest::gamg::GAMGMatrix referenceMatrix(4,owners,neighbours,diagonal,
                                                      Foam::scalarField{-0.2,0.0,-0.4});
        gsExperiment::Minimal4Data shared(referenceMatrix,true);
        Foam::scalarField rhs(4,1.0), expected(4,0.0), actual(5,0.0);
        smootherTest::gamg::gaussSeidelSmooth(expected,referenceMatrix,rhs,192);
        gsExperiment::minimal4<true>(actual,original,rhs,192,shared);
        if (std::memcmp(expected.data(),actual.data(),4*sizeof(double)) || actual[4]!=0.0)
            throw std::runtime_error("Padded4 gather/scatter storage mismatch");
    }
    for (int length=0; length<=32; ++length) {
        const int n=length+2;
        Foam::labelField owners, neighbours;
        Foam::scalarField coeff;
        for(int i=0;i<length;++i) {owners.push_back(0);neighbours.push_back(i+1);coeff.push_back(i%3==0?0.0:-0.125);}
        smootherTest::gamg::GAMGMatrix m(n,owners,neighbours,Foam::scalarField(n,4.0),coeff);
        Foam::scalarField rhs(n,1.0), baseline(n,0.0), reload(baseline),window(baseline);
        gsExperiment::Minimal4Data minimalData(m,false), paddedData(m,true);
        gsExperiment::MinimalOptData optData(m);
        for (int cell=0; cell<optData.terminalRow; ++cell)
            for (int face=m.ownerStartAddr()[cell]; face<m.ownerStartAddr()[cell+1]; face+=4)
                if (face+3>=m.nFaces())
                    throw std::runtime_error("ordinary load would cross allocated LDU end");
        auto optimized=baseline;
        gsExperiment::optimizedMinimal4(optimized,m,rhs,9,optData);
        auto minimal=baseline;
        auto padded=baseline;
        padded.push_back(0.0);
        gsExperiment::minimal4<false>(minimal,m,rhs,9,minimalData);
        gsExperiment::minimal4<true>(padded,m,rhs,9,paddedData);
        smootherTest::gamg::gaussSeidelSmooth(baseline,m,rhs,9);
        if (std::memcmp(optimized.data(),baseline.data(),n*sizeof(double)))
            throw std::runtime_error("optimized MinimalReload4 differs bitwise");
        if (std::memcmp(baseline.data(),minimal.data(),n*8)||std::memcmp(baseline.data(),padded.data(),n*8)||padded[n]!=0.0)
            throw std::runtime_error("minimal/padded mismatch or sentinel changed");
        gsExperiment::smooth<false>(reload,m,rhs,9);
        gsExperiment::smooth<true>(window,m,rhs,9);
        if (std::memcmp(baseline.data(),reload.data(),n*8)||std::memcmp(baseline.data(),window.data(),n*8)) throw std::runtime_error("bitwise mismatch");
    }
    for (int seed=1; seed<=12; ++seed) {
        const int n=47;
        Foam::labelField owners, neighbours;
        Foam::scalarField coeff;
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j)
            if (((i*37+j*19+seed*11)%17)<4) {
                owners.push_back(i); neighbours.push_back(j);
                coeff.push_back((j%3-1)*0.0625);
            }
        smootherTest::gamg::GAMGMatrix m(n,owners,neighbours,Foam::scalarField(n,8.0),coeff);
        Foam::scalarField rhs(n,1.0),baseline(n,-0.0),reload(baseline),window(baseline);
        gsExperiment::Minimal4Data minimalData(m,false), paddedData(m,true);
        gsExperiment::MinimalOptData optData(m);
        for (int cell=0; cell<optData.terminalRow; ++cell)
            for (int face=m.ownerStartAddr()[cell]; face<m.ownerStartAddr()[cell+1]; face+=4)
                if (face+3>=m.nFaces())
                    throw std::runtime_error("ordinary load would cross allocated LDU end");
        auto optimized=baseline;
        gsExperiment::optimizedMinimal4(optimized,m,rhs,9,optData);
        auto minimal=baseline;
        auto padded=baseline;
        padded.push_back(0.0);
        gsExperiment::minimal4<false>(minimal,m,rhs,9,minimalData);
        gsExperiment::minimal4<true>(padded,m,rhs,9,paddedData);
        smootherTest::gamg::gaussSeidelSmooth(baseline,m,rhs,9);
        if (std::memcmp(optimized.data(),baseline.data(),n*sizeof(double)))
            throw std::runtime_error("optimized MinimalReload4 differs bitwise");
        if (std::memcmp(baseline.data(),minimal.data(),n*8)||std::memcmp(baseline.data(),padded.data(),n*8)||padded[n]!=0.0)
            throw std::runtime_error("minimal/padded mismatch or sentinel changed");
        gsExperiment::smooth<false>(reload,m,rhs,9);
        gsExperiment::smooth<true>(window,m,rhs,9);
        if (std::memcmp(baseline.data(),reload.data(),n*8)||std::memcmp(baseline.data(),window.data(),n*8))
            throw std::runtime_error("multi-row bitwise mismatch");
    }
    std::cout << "row lengths 0..32, nine sweeps: bitwise equal\n";
}
