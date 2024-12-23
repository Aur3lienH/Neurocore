#pragma once
#include <iostream>


template<typename T>
class Vector2
{
public:
    Vector2(T x, T y)
    {
        this->x = x;
        this->y = y;
    }
    double& operator[](int index)
    {
        if (index == 0)
        {
            return this->x;
        }
        else if (index == 1)
        {
            return this->y;
        }
        else
        {
            std::cout << "Index out of bounds\n";
            return this->x;
        }
    }
private:
    T x;
    T y;
};

template<typename T>
class Vector3
{
public:
    Vector3(T x, T y , T z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    double& operator[] (int index)
    {
        if (index == 0)
        {
            return this->x;
        }
        else if (index == 1)
        {
            return this->y;
        }
        else if (index == 2)
        {
            return this->z;
        }
        else
        {
            std::cout << "Index out of bounds\n";
            return this->x;
        }
    }
private:
    T x;
    T y;
    T z;
};

template<typename T>
class Vector4
{
public:
    Vector4(T x, T y, T z, T w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    double& operator[] (int index)
    {
        if (index == 0)
        {
            return this->x;
        }
        else if (index == 1)
        {
            return this->y;
        }
        else if (index == 2)
        {
            return this->z;
        }
        else if (index == 3)
        {
            return this->w;
        }
        else
        {
            std::cout << "Index out of bounds\n";
            return this->x;
        }
    }
private:
    T x;
    T y;
    T z;
    T w;
};


