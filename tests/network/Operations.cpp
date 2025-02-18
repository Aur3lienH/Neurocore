#include "network/Operations.h"
#include <immintrin.h>

Add::Add(float* a, float* b,float* out, unsigned int length)
{
    this->a = a;
    this->b = b;
    this->out = out;
    this->length = length;
}


void Add::Compute()
{
    int i;
    for (i = 0; i + 8 <= length; i += 8)
    {
        __m256 m_a = _mm256_loadu_ps(a + i);
        __m256 m_b = _mm256_loadu_ps(b + i);
        __m256 m_out = _mm256_add_ps(m_a,m_b);
        _mm256_store_ps(out + i,m_out);
    }

    for (; i + 4 <= length; i += 4)
    {
        __m128 m_a = _mm_loadu_ps(a + i);
        __m128 m_b = _mm_loadu_ps(b + i);
        __m128 m_out = _mm_add_ps(m_a,m_b);
        _mm_store_ps(out + i, m_out);
    }

    for (; i < length; i++)
    {
        out[i] = a[i] + b[i];
    }
}

Mul::Mul(float* a, float* b, float* out, unsigned int length)
{
    this->a = a;
    this->b = b;
    this->out = out;
    this->length;
}


void Mul::Compute()
{
    int i;
    for (i = 0; i + 8 <= length; i += 8)
    {
        __m256 m_a = _mm256_loadu_ps(a + i);
        __m256 m_b = _mm256_loadu_ps(b + i);
        __m256 m_out = _mm256_mul_ps(m_a,m_b);
        _mm256_store_ps(out + i,m_out);
    }

    for (; i + 4 <= length; i += 4)
    {
        __m128 m_a = _mm_loadu_ps(a + i);
        __m128 m_b = _mm_loadu_ps(b + i);
        __m128 m_out = _mm_mul_ps(m_a,m_b);
        _mm_store_ps(out + i, m_out);
    }

    for (; i < length; i++)
    {
        out[i] = a[i] * b[i];
    }
}


MulAddTo1::MulAddTo1(float* a, float* b, float* output, unsigned int length)
{
    this->a = a;
    this->b = b;
    this->output = output;
    this->length = length;
}


void MulAddTo1::Compute()
{
    int i;
    for (i = 0; i + 8 <= length; i += 8)
    {
        __m256 m_a = _mm256_loadu_ps(a + i);
        __m256 m_b = _mm256_loadu_ps(b + i);
        __m256 m_out = _mm256_mul_ps(m_a,m_b);
        m_out = _mm256_hadd_ps(m_out, m_out);
        m_out = _mm256_hadd_ps(m_out, m_out);
        float temp[8];
        _mm256_store_ps(temp,m_out);
        *output += temp[0] + temp[1];
    }

    for (; i + 4 <= length; i += 4) {
        __m128 m_a = _mm_loadu_ps(a + i);
        __m128 m_b = _mm_loadu_ps(b + i);
        __m128 m_out = _mm_mul_ps(m_a, m_b);

        m_out = _mm_hadd_ps(m_out, m_out);
        m_out = _mm_hadd_ps(m_out, m_out);

        float temp[4];
        _mm_storeu_ps(temp, m_out);
        *output += temp[0];
    }

    for (; i < length; i++)
    { 
        *output += a[i] * b[i];
    }
}




EqualTo::EqualTo(float* a, float number, unsigned int length)
{
    this->a = a;
    this->number = number;
    this->length = length;
}

void EqualTo::Compute()
{
    int i;
    for (i = 0; i + 8 <= length; i += 8)
    {
        __m256 temp = _mm256_set1_ps(number);
        _mm256_storeu_ps(a + i,temp);
    }
    for (; i + 4 <= length; i+=4)
    {
        __m128 temp = _mm_set1_ps(number);
        _mm_storeu_ps(a + i, temp);
    }
    for (; i < length; i++)
    {
        a[i] = number;
    }
}