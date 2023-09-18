#pragma once
#include "Bytes.h"
#include <vector>
#include <iostream>

namespace Tools
{
    int WriteUnsignedChar(const std::string& filename, unsigned char* bytes, int size);

    Bytes ReadUnsignedChar(const std::string& filename);

    int WriteText(const std::string& filename, const std::string& text);

    std::string ReadText(const std::string& filename);

    int ContinueWritingText(const std::string& filename, const std::string& text);

    int CreatePathDirectory(const std::string& path);

    int CreatePathFile(const std::string& filePath);

    int StringToBytes(std::string str, unsigned char* bytes);

    std::string BytesToString(unsigned char* bytes, int* length);

    int IntToBytes(int value, unsigned char* bytes);

    int BytesToInt(const unsigned char* bytes);

    template<typename T>
    int VectorToBytes(std::vector<T> values, unsigned char* buf)
    {
        int length = sizeof(int);
        IntToBytes(values.size(), buf);
        for (int i = 0; i < values.size(); i++)
        {
            if (std::is_same<int, T>())
            {
                length = IntToBytes(values[i], buf + length);
            }
            else if (std::is_same<float, T>())
            {
                memcpy(buf + length, &values[i], sizeof(float));
                length += sizeof(float);
            }
            else if (std::is_same<double, T>())
            {
                memcpy(buf + length, &values[i], sizeof(double));
                length += sizeof(double);
            }
            else if (std::is_same<char, T>())
            {
                mempcpy(buf + length, &values[i], sizeof(char));
                length += sizeof(char);
            }
            else if (std::is_same<unsigned char, T>())
            {
                memcpy(buf + length, &values[i], sizeof(unsigned char));
                length += sizeof(unsigned char);
            }
            else if (std::is_same<bool, T>())
            {
                memcpy(buf + length, &values[i], sizeof(bool));
                length += sizeof(bool);
            }
            else if (std::is_same<std::string, T>())
            {
                length += StringToBytes(values[i], buf + length);
            }
            else
            {
                std::cout << "Type not supported ! \n";
                return -1;
            }
        }
        return length;
    }
}

