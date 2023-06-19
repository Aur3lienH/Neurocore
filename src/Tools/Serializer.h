#pragma once
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "ManagerIO.h"




namespace Tools
{
    class Data
    {
    public:
        Data(ulong _size, void* pointer);
        ulong size;
        void* pointer;
    };
    


    class Serializer
    {
    public:
        Serializer(std::size_t count, ...);
        void Save(std::ofstream& writer);
        void Load(std::ifstream& reader);
    private:
        std::vector<Data*> datas;
    };

}
