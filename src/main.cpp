#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#include "neurone.hpp"
#include "dataset.hpp"
#include "rand.hpp"
//#include "layer.hpp"

#include "network.hpp"


/*
std::vector<float> phoneToInput(const Phone& p) {
    return {
        p.battery,
        p.ram,
    };
}
*/

int main() {
    DataLoader df("../data/train1.csv");
    
   std::vector<Phone> phones = df.load();
   
    for(auto &i : phones) {
      Random rn(0, 1);
      float w1 = rn.randx<float>(0, 1);
      float w2 = rn.randx<float>(0, 1);
      float bs1 = rn.randx<float>(0, 1);
      float bs2 = rn.randx<float>(0, 1);
      
     // std::cout << ">" << w1<< "<" << w2 << "<" << bs << std::endl;
      Neurone node(i.battery, i.ram, w1, w2, bs1, bs2);
     //std::cout << "\n\n->" << i.price << "<--" << node.outputWeight() << std::endl;
     Network nn({2, 3, 2});
     /*
     std::vector<float> networkInput;
     networkInput.push_back(node.outputWeight());
     */
     int preds = nn.classify(node.outputWeight());
     std::cout << "  price: " << i.price <<  " pred: " << preds << " ba3: " << node.outputWeight()[0] << " ram:" << node.outputWeight()[1] << std::endl;
    }
    
    
    
}
