#pragma once

#include <iostream>

namespace Tools
{
    class Unit
    {
    public:
        explicit Unit(const std::string& _unitName);

        Unit(const std::string& _unitName, double _value);

        friend std::ostream& operator<<(std::ostream& os, const Unit& unit);

        ~Unit();

    private:
        std::string unitName;
        double value;
    };
}
