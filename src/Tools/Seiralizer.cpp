#include "Serializer.h"
#include <fstream>
#include <cstdarg>
#include <vector>


namespace Tools
{

    Data::Data(ulong _size, void* _pointer)
    {
        size = _size;
        pointer = _pointer;
    }


    Serializer::Serializer(std::size_t count, ...)
    {
        va_list valist;
        va_start(valist, count); 
        for (int i = 0; i < count; i++)
        {
            Data* buf = va_arg(valist, Data*);
            datas.push_back(buf);
        }
        va_end(valist);
    }



    void Serializer::Save(std::ofstream& writer)
    {
        for (int i = 0; i < datas.size(); i++)
        {
            if(datas[i]->size == 0)
            {
                Serializer* se = (Serializer*)datas[i]->pointer;
                se->Save(writer);
            }
            else
            {
                writer.write(reinterpret_cast<char*>(datas[i]->pointer),datas[i]->size);
            }
        }
    }

    void Serializer::Load(std::ifstream& reader)
    {
        for (int i = 0; i < datas.size(); i++)
        {
            if(datas[i]->size == 0)
            {
                Serializer* se = (Serializer*)datas[i]->pointer;
                se->Load(reader);
            }
            else
            {
                reader.read(reinterpret_cast<char*>(datas[i]->pointer),datas[i]->size);
            }
        }
    }
}
