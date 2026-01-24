#pragma once
#ifndef __NEURONE_H
#define __NEURONE_H
#include <vector>
#include <string>
#include <cmath>

class Neurone {
  public:
    Neurone(int x, int y, float w1, float w2, float bs1, float bs2) :
    input1(x),
    input2(y),
    weight1(w1),
    weight2(w2),
    bias1(bs1),
    bias2(bs2)
    {}
    
    std::vector<float> outputWeight() {
      float out1 = (weight1  * input1) + bias1;
      float out2 = (weight2 * input2) + bias2;
      return {out1, out2};
      /*
      if (out1 > out2) {
        return 0;
      } else {
        return 1;
      }
      */
    }
    
  private:
    int input1;
    int input2;
    float weight1;
    float weight2;
    float bias1;
    float bias2;
};
#endif