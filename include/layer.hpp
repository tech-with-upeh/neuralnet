#pragma once
#ifndef __LAYER_H
#define __LAYER_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
//#include <ctime>


struct Phone {
    float battery;
    float ram;
    int label;
};

struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    float& operator()(int i, int j) { return data[i * cols + j]; }
};

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float sigmoid_derivative(float y) { return y * (1.0f - y); }
float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }



class Layer {
  public:
    Layer(int in, int out, bool relu_layer) 
    : in(in), out(out), W(in, out), b(out, 0.0f), use_relu(relu_layer)
    {
        for (int i = 0; i < in; ++i)
            for (int o = 0; o < out; ++o)
                W(i,o) = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    
    int in, out;
    Matrix W;
    std::vector<float> b;
    std::vector<float> input, z, a;
    bool use_relu;

    std::vector<float> forward(const std::vector<float>& x) {
        input = x;
        z.assign(out, 0.0f);
        a.assign(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            z[o] = b[o];
            for (int i = 0; i < in; ++i)
                z[o] += x[i] * W(i,o);
            a[o] = use_relu ? relu(z[o]) : sigmoid(z[o]);
        }
        return a;
    }

    std::vector<float> backward(const std::vector<float>& grad, float lr) {
        std::vector<float> grad_input(in, 0.0f);
        for (int o = 0; o < out; ++o) {
            float dz = grad[o] * (use_relu ? relu_derivative(z[o]) : sigmoid_derivative(a[o]));
            for (int i = 0; i < in; ++i) {
                grad_input[i] += W(i,o) * dz;
                W(i,o) -= lr * dz * input[i];
            }
            b[o] -= lr * dz;
        }
        return grad_input;
    }
};

#endif