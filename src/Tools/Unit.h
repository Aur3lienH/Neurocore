#pragma once
#include <iostream>

namespace Tools
{
    class Unit
    {
    public:
        Unit(std::string unitName);
        Unit(std::string unitName, double value);
        friend std::ostream& operator<<(std::ostream& os, const Unit& unit);
        ~Unit();
    private:
        std::string unitName;
        double value;
    };
}
