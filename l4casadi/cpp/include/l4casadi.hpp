#include <iostream>
#include <experimental/propagate_const>

#ifndef l4casadi_hpp
#define l4casadi_hpp

class L4CasADi
{
public:
    L4CasADi(std::string, std::string, std::string = "cpu");
    ~L4CasADi();
    void forward(const double*, int, int, double*);
    void jac(const double*, int, int, double*);

    // PImpl Idiom
    class L4CasADiImpl;
    std::experimental::propagate_const<std::unique_ptr<L4CasADiImpl> > pImpl;

};

#endif /* l4casadi_hpp */