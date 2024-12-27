#pragma once
#include "matrix/Matrix.cuh"

template<typename Derived>
class Loss final
{
public:
    Loss();

    ~Loss() = default;

    template<int rows, int cols, int dims>
    double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target)
    {
      return static_cast<Derived*>(this)->Cost(output, target);
    }

    template<int rows, int cols, int dims>
    void CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result)
    {
        static_cast<Derived*>(this)->CostDerivative(output, target, result);
    }

    //static Loss* Read(std::ifstream& reader);

    //void Save(std::ofstream& writer);

protected:
    int ID;
};

