#include <iostream>
#include <memory>

#ifndef l4casadi_hpp
#define l4casadi_hpp

class L4CasADi
{
private:
    int rows_in;
    int cols_in;

    int rows_out;
    int cols_out;
public:
    L4CasADi(std::string, std::string, int, int, int, int, std::string = "cpu", bool = false, bool = false, bool = false, bool = false,
        bool = false,bool = false);
    ~L4CasADi();
    void forward(const double*, double*);
    void jac(const double*, double*);
    void adj1(const double*, const double*, double*);
    void jac_adj1(const double*, const double*, double*);
    void jac_jac(const double*, double*);

    void invalid_argument(std::string);

    // PImpl Idiom
    class L4CasADiImpl;
    class L4CasADiScriptedImpl;
    class L4CasADiCompiledImpl;
    std::unique_ptr<L4CasADiImpl> pImpl;

};

#endif /* l4casadi_hpp */