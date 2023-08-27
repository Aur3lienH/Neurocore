#include "Serializer.cuh"
#include <fstream>
#include <cstdarg>


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
        for (auto& data : datas)
        {
            if (data->size == 0)
            {
                auto* se = (Serializer*) data->pointer;
                se->Save(writer);
            }
            else
            {
                writer.write(reinterpret_cast<char*>(data->pointer), data->size);
            }
        }
    }

    void Serializer::Load(std::ifstream& reader)
    {
        for (auto& data : datas)
        {
            if (data->size == 0)
            {
                auto* se = (Serializer*) data->pointer;
                se->Load(reader);
            }
            else
            {
                reader.read(reinterpret_cast<char*>(data->pointer), data->size);
            }
        }
    }
}
