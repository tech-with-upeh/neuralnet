#pragma once
#ifndef __RAND_H
#define __RAND_H
#include <random>
#include <chrono>

class Random {
  public:
    Random(int min, int max):
    min_(min),
    max_(max) {}
    
    template <typename T>
    T rand() {
      int rn;
      std::random_device rd;
      std::seed_seq ss{static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()), rd(), rd(), rd()};
      std::mt19937_64 rng(ss);
      std::uniform_int_distribution<T> dist(min_,max_);
      if constexpr (std::is_integral_v<T>) {
        //std::uniform_int_distribution<T> dist(min_,max_);
      } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_,max_);
      } else {
        static_assert(std::is_arithmetic_v<T>,
            "T must be an arithmetic type");
      }
      return dist(rng);
    }
    
    // Normal distribution helper
int normal_rand(int mean, int stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);
    return std::round(d(gen));
}
    
    template <typename T>
    T randx(int x, int y) {
      std::random_device rd;
      std::seed_seq ss{static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()), rd(), rd(), rd()};
      
      std::mt19937_64 rng(ss);
      std::uniform_real_distribution<T> dist(x,y);
      if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(x,y);
      } else if constexpr (std::is_floating_point_v<T>) {
        //std::uniform_real_distribution<T> dist(x,y);
        
      } else {
        static_assert(std::is_arithmetic_v<T>,
            "T must be an arithmetic type");
      }
      
      return dist(rng);
    }
    
  private:
    int min_;
    int max_;
};

#endif