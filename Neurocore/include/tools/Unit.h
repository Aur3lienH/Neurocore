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

namespace Tools
{
    Unit::Unit(const std::string& _unitName)
    {
        unitName = _unitName;
        value = 0;
    }

    Unit::Unit(const std::string& _unitName, double _value)
    {
        unitName = _unitName;
        value = _value;
    }

    Unit::~Unit()
    = default;

    std::ostream& operator<<(std::ostream& os, const Unit& unit)
    {
        double outputValue = unit.value;
        std::string unitExtension;

        if (outputValue > 1000000000000000)
        {
            outputValue /= 1000000000000000;
            unitExtension = "P";
        }
        else if (outputValue > 1000000000000)
        {
            outputValue /= 1000000000000;
            unitExtension = "T";
        }
        else if (outputValue > 1000000000)
        {
            outputValue /= 1000000000;
            unitExtension = "G";
        }
        else if (outputValue > 1000000)
        {
            outputValue /= 1000000;
            unitExtension = "M";
        }
        else if (outputValue > 1000)
        {
            outputValue /= 1000;
            unitExtension = "K";
        }
        else if (outputValue > 1)
        {
            unitExtension = "";
        }
        else if (outputValue > 1e-3)
        {
            outputValue *= 1000;
            unitExtension = "m";
        }
        else if (outputValue > 1e-6)
        {
            outputValue *= 1e+6;
            unitExtension = "μ";
        }
        else if (outputValue > 1e-9)
        {
            outputValue *= 1e+9;
            unitExtension = "n";
        }
        else
        {
            outputValue *= 1e+12;
            unitExtension = "p";
        }
        os << outputValue << " " << unitExtension << unit.unitName;
        return os;
    }
}