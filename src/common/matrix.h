// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: matrix

#ifndef __CUDA_HGEMM_MATRIX_H__
#define __CUDA_HGEMM_MATRIX_H__

#include <random>

#include "common.h"

class Matrix {
public:
    Matrix(size_t row, size_t col, const std::string &name = "Matrix", float min = -2.0, float max = 2.0)
        : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
        HGEMM_CHECK_GT(m_row, 0);
        HGEMM_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        HGEMM_CHECK_GT(m_elem_num, 0);

        m_data = new half[m_elem_num];
        HGEMM_CHECK(m_data);
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_gpu_data, m_elem_num * sizeof(half)));
        HGEMM_CHECK(m_gpu_data);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_data[i] = __float2half(uniform(engine));
        }

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_gpu_data, m_data, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));

        HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_data, m_gpu_data);
    }

    ~Matrix() {
        if (m_data) {
            delete[] m_data;
            m_data = nullptr;
        }

        if (m_gpu_data) {
            HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_gpu_data));
            m_gpu_data = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    half *getData() const {
        return m_data;
    }

    half *getGpuData() const {
        return m_gpu_data;
    }

    void zeros() {
        HGEMM_CHECK_CUDART_ERROR(
            cudaMemset(m_gpu_data, 0x00, m_elem_num * sizeof(half)));
        moveToHost();
    }

    void random(float min = -2.0, float max = 2.0) {
        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(min, max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_data[i] = __float2half(uniform(engine));
        }
        moveToHost();
    }

    void tearUp(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        HGEMM_CHECK_CUDART_ERROR(
            cudaMemcpy(m_gpu_data, base->getData(), m_elem_num * sizeof(half), cudaMemcpyHostToDevice));
    }

    void moveToHost() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_data, m_gpu_data, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost));
    }

    void checkValue(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        float diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_data[i]) - __half2float(base->getData()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= double(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    const float m_min = -2.0;
    const float m_max = 2.0;

    size_t m_elem_num = 0;
    half *m_data = nullptr;
    half *m_gpu_data = nullptr;

    float m_max_diff = 0.0;
    float m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};

#endif  // __CUDA_HGEMM_MATRIX_H__
