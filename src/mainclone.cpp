#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>

/* ================= MATRIX ================= */
struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    float& operator()(int i, int j) { return data[i * cols + j]; }
};

/* ================= ACTIVATIONS ================= */
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float sigmoid_derivative(float y) { return y * (1.0f - y); }
float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

/* ================= DENSE LAYER ================= */
class Dense {
public:
    Dense(int in, int out, bool relu_layer)
        : in(in), out(out), W(in, out), b(out, 0.0f), use_relu(relu_layer)
    {
        for (int i = 0; i < in; ++i)
            for (int o = 0; o < out; ++o)
                W(i,o) = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

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

private:
    int in, out;
    Matrix W;
    std::vector<float> b;
    std::vector<float> input, z, a;
    bool use_relu;
};

/* ================= PHONE DATA ================= */
struct Phone {
    float battery;
    float ram;
    int label;
};

/* ================= CSV LOADER ================= */
std::vector<Phone> loadCSV(const std::string& file) {
    std::vector<Phone> data;
    std::ifstream f(file);
    std::string line;
    if (!f) { std::cerr << "Cannot open " << file << "\n"; return data; }
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string item;
        Phone p;
        std::getline(ss, item, ','); p.battery = std::stof(item);
        std::getline(ss, item, ','); p.ram = std::stof(item);
        std::getline(ss, item, ','); p.label = std::stoi(item);
        data.push_back(p);
    }
    return data;
}

/* ================= LOSS ================= */
float binary_cross_entropy(float y, float y_hat) {
    y_hat = std::fmax(1e-7f, std::fmin(1.0f - 1e-7f, y_hat));
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat));
}

/* ================= NETWORK CLASS ================= */
class Network {
public:
    // layers_spec: vector of {num_inputs, num_outputs} for each layer
    Network(const std::vector<std::pair<int,int>>& layers_spec) {
        for (size_t i = 0; i < layers_spec.size(); ++i) {
            int in = layers_spec[i].first;
            int out = layers_spec[i].second;
            bool use_relu = (i != layers_spec.size() - 1); // last layer = sigmoid
            layers.emplace_back(Dense(in, out, use_relu));
        }
    }

    std::vector<float> forward(const std::vector<float>& x) {
        std::vector<float> input = x;
        for (auto& layer : layers)
            input = layer.forward(input);
        return input;
    }

    void train(const std::vector<Phone>& data, int epochs, float lr) {
        for (int e = 0; e < epochs; ++e) {
            float loss = 0.0f;
            int correct = 0;

            for (const Phone& p : data) {
                std::vector<float> x = { p.battery / 2000.0f - 0.5f, p.ram / 4000.0f - 0.5f };

                // FORWARD PASS
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

                // BACKWARD PASS
                std::vector<float> grad = { y_hat - p.label };
                for (int i = layers.size() - 1; i >= 0; --i)
                    grad = layers[i].backward(grad, lr);
            }

            if (e % 500 == 0) {
                float acc = (float)correct / data.size() * 100.0f;
                std::cout << "[TRAIN] Epoch " << e
                          << " | Loss: " << loss / data.size()
                          << " | Acc: " << acc << "%\n";
            }
        }
    }

    void test(const std::vector<Phone>& data) {
        int correct = 0;
        std::cout << "\n--- TEST RESULTS ---\n";
        for (const Phone& p : data) {
            std::vector<float> x = { p.battery / 2000.0f - 0.5f, p.ram / 4000.0f - 0.5f };
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
    std::vector<Dense> layers;
};

//network feom network.cpp
class Network {
public:
    Network(const std::vector<std::pair<int,int>>& layers_spec) {
        for (size_t i = 0; i < layers_spec.size(); ++i) {
            int in = layers_spec[i].first;
            int out = layers_spec[i].second;
            bool use_relu = (i != layers_spec.size() - 1); // last layer = sigmoid
            layers.emplace_back(Layer(in, out, use_relu));
        }
    }

    std::vector<float> forward(const std::vector<float>& x) {
        std::vector<float> input = x;
        for (auto& layer : layers)
            input = layer.forward(input);
        return input;
    }

    void train(const std::vector<Phone>& data, int epochs, float lr) {
        for (int e = 0; e < epochs; ++e) {
            float loss = 0.0f;
            int correct = 0;

            for (const Phone& p : data) {
                std::vector<float> x = { p.battery / 1997.0f - 0.5f, p.ram / 3998.0f - 0.5f };

                // FORWARD PASS
                std::vector<std::vector<float>> activations;
                activations.push_back(x);
                std::vector<float> input = x;
                for (auto& layer : layers) {
                    input = layer.forward(input);
                    activations.push_back(input);
                }
                
                if(activations.empty()) {
                  std::cerr << "Activations vector was empty! , will break network" << std::endl;
                  exit(1);
                }
                float y_hat = activations.back()[0];
                loss += binary_cross_entropy(p.label, y_hat);

                int pred = y_hat >= 0.5f ? 1 : 0;
                if (pred == p.label) correct++;

                //back prop
                std::vector<float> grad = { y_hat - p.label };
                for (int i = layers.size() - 1; i >= 0; --i)
                    grad = layers[i].backward(grad, lr);
            }

            if (e % 500 == 0) {
                float acc = (float)correct / data.size() * 100.0f;
                std::cout << "[TRAIN] Epoch " << e
                          << " | Loss: " << loss / data.size()
                          << " | Acc: " << acc << "%\n";
            }
        }
    }

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


/* ================= MAIN ================= */
int main() {
    srand((unsigned)time(nullptr));

    auto train_data = loadCSV("../data/train1.csv");
    auto test_data  = loadCSV("../data/train2.csv");

    // Example flexible architecture: 2 inputs → 4 hidden → 1 output
    std::vector<std::pair<int,int>> arch = { {2,8}, {8,4}, {4,1} };
    Network network(arch);

    network.train(train_data, 5000, 0.02f);
    network.test(test_data);

    return 0;
}
