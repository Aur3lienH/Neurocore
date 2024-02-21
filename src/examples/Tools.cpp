#include "examples/Tools.h"
#include <iostream>
#include <fstream>


int CSVTools::CsvLength(const std::string& path)
{
    std::ifstream file(path);
    std::string line;

    int i = 0;
    if (file.is_open())
    {
        while (getline(file, line))
        {
            i++;
        }
    }
    else
    {
        throw std::runtime_error("File not found");
    }

    return i;

}