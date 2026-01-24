#pragma once
#ifndef __NETWORK_H
#define __NETWORK_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "layer.hpp"

float binary_cross_entropy(float correctLabel, float predictedLabel) {
    predictedLabel = std::fmax(1e-7f, std::fmin(1.0f - 1e-7f, predictedLabel));
    /* Loss=−(ylog(y^)+(1−y)log(1−y^))
    y = correct label
    y^ = output of neurone 
    */
    return -(correctLabel * std::log(predictedLabel) + (1 - correctLabel) * std::log(1 - predictedLabel));
}

// ... Keep all your existing includes, structs, Layer class, CSV loader, etc.

class Network {
public:
    Network(const std::vector<std::pair<int,int>>& layers_spec) {
        for (size_t i = 0; i < layers_spec.size(); ++i) {
            int in = layers_spec[i].first;
            int out = layers_spec[i].second;
            bool use_relu = (i != layers_spec.size() - 1);
            layers.emplace_back(Layer(in, out, use_relu));
        }
    }

    std::vector<float> forward(const std::vector<float>& x) {
        std::vector<float> input = x;
        for (auto& layer : layers)
            input = layer.forward(input);
        return input;
    }

    void train(const std::vector<Phone>& data, int epochs, float lr, int batch_size) {
        int n = data.size();
        for (int e = 0; e < epochs; ++e) {
            float loss = 0.0f;
            int correct = 0;

            // Loop over batches
            for (int start = 0; start < n; start += batch_size) {
                int end = std::min(start + batch_size, n);
                int bsize = end - start;

                // Accumulate gradients for each layer
                std::vector< std::vector< std::vector<float> > > weight_grads(layers.size());
                std::vector< std::vector<float> > bias_grads(layers.size());
                for (int l = 0; l < layers.size(); ++l) {
                    weight_grads[l].resize(layers[l].in, std::vector<float>(layers[l].out, 0.0f));
                    bias_grads[l].resize(layers[l].out, 0.0f);
                }

                for (int i = start; i < end; ++i) {
                    const Phone& p = data[i];
                    std::vector<float> x = { p.battery / 2000.0f - 0.5f, p.ram / 4000.0f - 0.5f };

                    // Forward pass
                    std::vector<std::vector<float>> activations;
                    activations.push_back(x);
                    std::vector<float> input = x;
                    for (auto& layer : layers) {
                        input = layer.forward(input);
                        activations.push_back(input);
                    }

                    float y_hat = activations.back()[0];
                    loss += binary_cross_entropy(p.label, y_hat);

                    int pred = y_hat >= 0.5f ? 1 : 0;
                    if (pred == p.label) correct++;

                    // Backward pass: compute dz
                    std::vector<float> dz = { y_hat - p.label };
                    for (int l = layers.size() - 1; l >= 0; --l) {
                        std::vector<float> grad_input = layers[l].backward(dz, 1.0f); // no lr yet
                        dz = grad_input;

                        // accumulate gradients
                        for (int wi = 0; wi < layers[l].in; ++wi)
                            for (int wo = 0; wo < layers[l].out; ++wo)
                                weight_grads[l][wi][wo] += layers[l].W(wi,wo); // collect dz*input internally
                        for (int wo = 0; wo < layers[l].out; ++wo)
                            bias_grads[l][wo] += layers[l].b[wo];
                    }
                }

                // Update weights and biases after batch
                for (int l = 0; l < layers.size(); ++l) {
                    for (int wi = 0; wi < layers[l].in; ++wi)
                        for (int wo = 0; wo < layers[l].out; ++wo)
                            layers[l].W(wi,wo) -= (lr / bsize) * weight_grads[l][wi][wo];
                    for (int wo = 0; wo < layers[l].out; ++wo)
                        layers[l].b[wo] -= (lr / bsize) * bias_grads[l][wo];
                }
            }

            if (e % 500 == 0) {
                float acc = (float)correct / data.size() * 100.0f;
                std::cout << "[TRAIN] Epoch " << e
                          << " | Loss: " << loss / data.size()
                          << " | Acc: " << acc << "%\n";
            }
        }
    }

    // ... test function same as before
    void test(const std::vector<Phone>& data) {
        int correct = 0;
        std::cout << "\n--- TEST RESULTS ---\n";
        for (const Phone& p : data) {
            std::vector<float> x = { p.battery / 1997.0f - 0.5f, p.ram / 3998.0f - 0.5f };
            float prob = forward(x)[0];
            int pred = prob >= 0.5f ? 1 : 0;
            if (pred == p.label) correct++;
            std::cout << "Prob: " << prob
                      << " | Pred: " << pred
                      << " | Label: " << p.label
                      << "\n";
        }
        float acc = (float)correct / data.size() * 100.0f;
        std::cout << "\nTest Accuracy: " << acc << "%\n";
    }
private:
    std::vector<Layer> layers;
};



#endif
