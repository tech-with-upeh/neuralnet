#pragma once
#ifndef __DATASETLOADER_H
#define __DATASETLOADER_H
// #include <iostream>
#include <vector>
#include <string>
#include <sstream>

struct Phone {
  int battery;
  int ram;
  int price;
  int model;
};

class DataLoader {
  public:
    DataLoader(std::string _file) : 
      filename(_file) {}
      
    std::vector<Phone> load() {
      std::vector<Phone> phones;
      std::ifstream file(filename);
      
      if(!file.is_open()) {
        std::cerr << "couldnt open " << std::endl;
        return phones;
      }
      std::string line;
  
      std::getline(file, line); // skip ader
  
      while (std::getline(file, line)) {
          std::stringstream ss(line);
          Phone p;
          std::string temp;
          
          /*
          ba3 - 502 - 1997
          ram - 256 - 3998
          */
  
          std::getline(ss, temp, ',');
          p.battery = (std::stoi(temp) -502) / (1997 - 502);
          
          std::getline(ss, temp, ',');
          p.ram = (std::stoi(temp) - 256) / (3998 - 256);
          
          std::getline(ss, temp, ',');
          p.price = std::stoi(temp);
          phones.push_back(p);
      }
      
      return phones;
    }
    
  private:
    std::string filename;
};
#endif