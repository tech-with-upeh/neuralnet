#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "layer.hpp"
#include "network.hpp"



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


/* ================= MAIN ================= */
int main() {
    srand((unsigned)time(nullptr));

    auto train_data = loadCSV("../data/train1.csv");
    auto test_data  = loadCSV("../data/train2.csv");

    // Example flexible architecture: 2 inputs → 4 hidden → 1 output
    std::vector<std::pair<int,int>> arch = { {2,8}, {8,4}, {4,1} };
    Network network(arch);

    network.train(train_data, 5000, 0.02f, 50);
    network.test(test_data);

    return 0;
}
