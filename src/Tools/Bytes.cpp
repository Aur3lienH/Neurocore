#include <iostream>
#include "Bytes.h"

namespace Tools
{
	unsigned char& Bytes::operator[](int index)
	{
		if (index >= length || index < 0)
		{
			std::cout << "Array index out of the bound, existing !";
		}
		return bytes[index];
	}
	Bytes::Bytes(unsigned char* _bytes, uint64_t _length)
	{
		bytes = _bytes;
		length = _length;
	}
	uint64_t Bytes::Length()
	{
		return length;
	}
	unsigned char* Bytes::GetBytes()
	{
		return bytes;
	}

}