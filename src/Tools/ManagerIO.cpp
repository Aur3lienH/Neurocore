#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "ManagerIO.h"
namespace Tools
{
    int WriteUnsignedChar(std::string filename,unsigned char * bytes,int size)
    {
        std::ofstream out;
        out.open(filename, std::ios::binary);
        if(!out)
        {
            std::cout << "Cannot open File ! \n";
            return -1;
        }
        out.flush();
        out.write((char*)bytes,size);
        
        out.close();
        return 1;
    }

    Bytes ReadUnsignedChar(std::string filename)
    {
        std::ifstream in;
        in.open(filename, std::ios::binary);
        in.seekg(0,std::ios::end);
        int length = in.tellg();
        in.seekg(0,std::ios::beg);
        unsigned char* buffer = new unsigned char[length];
        in.read((char*)buffer,length);
        
        in.close();
        return Bytes(buffer,length);
    }


    int WriteText(std::string filename, std::string text)
    {
        std::ofstream out;
        out.open(filename);
        out << text;
        out.close();
        return 1;
    }

    std::string ReadText(std::string filename)
    {
        std::ifstream in;
        in.open(filename);
        std::string output;
        in >> output;
        in.close();
        return output;
    }
    int ContinueWritingText(std::string filename, std::string text)
    {
        std::ofstream out;
        out.open(filename, std::ios::app);
        out << text;
        out.close();
        return 1;
    }
    int CreatePathDirectory(std::string path)
    {
        std::vector<std::filesystem::path> pathVector;
        std::filesystem::path p(path);
        std::filesystem::path tempPath = p;
        while(tempPath.string().size() > 0)
        {

            pathVector.push_back(tempPath);
            tempPath = tempPath.parent_path();
            
        }
        for (int i = pathVector.size() - 1; i >= 0; i--)
        {
            if(!std::filesystem::is_directory(pathVector[i]))
            {
                std::cout << pathVector[i]<< "\n";
                if(std::filesystem::create_directory(pathVector[i]) != 1)
                {
                    std::cout << "Cannot create directory ! \n";
                    return -1;
                }
            }
        }
        return 1;
    }

    int CreatePathFile(std::string filePath)
    {
        std::vector<std::filesystem::path> pathVector;
        std::filesystem::path p(filePath);
        std::filesystem::path tempPath = p;
        tempPath = tempPath.parent_path();
        while(tempPath.string().size() > 0)
        {

            pathVector.push_back(tempPath);
            tempPath = tempPath.parent_path();
            
        }
        for (int i = pathVector.size() - 1; i >= 0; i--)
        {
            if(!std::filesystem::is_directory(pathVector[i]))
            {
                std::cout << pathVector[i]<< "\n";
                if(std::filesystem::create_directory(pathVector[i]) != 1)
                {
                    std::cout << "Cannot create directory ! \n";
                    return -1;
                }
            }
        }
        return 1;
    }

    int StringToBytes(std::string str, unsigned char* buf)
    {
        int length = str.size() * sizeof(char) + sizeof (int);
        IntToBytes(str.size(), buf);
        for (int i = 0; i < str.size(); i++)
        {
            buf[i + sizeof(int)] = (unsigned char)str[i];
        }
        return length;
    }

    std::string BytesToString(unsigned char* bytes,int* length)
    {
        int size = BytesToInt(bytes);
        std::string str;
        for (int i = 0; i < size; i++)
        {
            str.push_back(bytes[i + sizeof(int)]);
        }
        *length += sizeof(int) + str.size() * sizeof(char);
        return str;
    }

    int IntToBytes(int value, unsigned char* bytes)
    {
        bytes[0] = (value >> 24) & 0xFF;
        bytes[1] = (value >> 16) & 0xFF;
        bytes[2] = (value >> 8) & 0xFF;
        bytes[3] = value & 0xFF;
        return 4;
    }

    int BytesToInt(unsigned char* bytes)
    {
        int value = 0;
        value = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        return value;
    }

}
