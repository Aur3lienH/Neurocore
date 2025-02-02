#pragma once

#include <iostream>
#include <cstdint>
namespace Tools
{
    class Bytes
    {
    public:
        Bytes(unsigned char* _bytes, uint64_t _length);

        uint64_t Length();

        unsigned char* GetBytes();

        unsigned char& operator[](int);

    private:
        unsigned char* bytes;
        uint64_t length;
    };
}

