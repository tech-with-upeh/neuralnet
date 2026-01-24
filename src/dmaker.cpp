#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

std::vector<std::string> fems = {
    "Adebimpe","Adesola","Damilola","Funke","Bolanle","Yewande","Abisola",
    "Morayo","Oluwafunke","Ronke","Sade","Chiamaka","Ifunanya","Nkiru",
    "Ngozi","Chibuzo","Uchechi","Obianuju","Amaka","Ijeoma","Chizoba",
    "Aisha","Zainab","Hauwa","Fatima","Safiya","Maryam","Hadiza","Sadiya",
    "Rahma","Khadija","Blessing","Peace","Joy","Esther","Deborah","Ruth","Grace"
};

std::vector<std::string> males = {
    "Ade","Adebayo","Adebola","Adedayo","Oluwaseun","Olawale","Babajide",
    "Damilola","Opeyemi","Taiwo","Kehinde","Chinedu","Emeka","Ifeanyi",
    "Nnamdi","Chukwuemeka","Uche","Obinna","Onyekachi","Ikenna","Chibuzo",
    "Sadiq","Musa","Abdul","Abdullahi","Ibrahim","Yusuf","Sule","Ismail",
    "Usman","Kabir","Samuel","David","Daniel","Michael","Joseph","Joshua","Isaac"
};

struct Person {
    std::string name;
    int height;
    int shoesize;
    int gender; // 0=female, 1=male
};

std::random_device rd;
std::mt19937 gen(rd());

// Randomly shuffle a vector
template<typename T>
void shuffle_vector(std::vector<T>& v) {
    std::shuffle(v.begin(), v.end(), gen);
}

// Generate a distinct dataset
int main() {
    std::ofstream file("distinct_dataset.csv");
    file << "Name,Height,ShoeSize,Gender\n";

    int n_female = 500;
    int n_male = 500;

    // Generate discrete ranges for females
    std::vector<int> female_heights;
    for(int h=150; h<=169; h++) female_heights.push_back(h); // 20 values
    std::vector<int> female_shoes;
    for(int s=35; s<=40; s++) female_shoes.push_back(s); // 6 values

    // Repeat to reach n_female, then shuffle
    while(female_heights.size() < n_female) female_heights.insert(female_heights.end(), female_heights.begin(), female_heights.end());
    while(female_shoes.size() < n_female) female_shoes.insert(female_shoes.end(), female_shoes.begin(), female_shoes.end());

    female_heights.resize(n_female);
    female_shoes.resize(n_female);

    shuffle_vector(female_heights);
    shuffle_vector(female_shoes);
    shuffle_vector(fems);

    for(int i=0; i<n_female; i++){
        file << fems[i % fems.size()] << "," << female_heights[i] << "," << female_shoes[i] << ",0\n";
    }

    // Generate discrete ranges for males
    std::vector<int> male_heights;
    for(int h=175; h<=194; h++) male_heights.push_back(h); // 20 values
    std::vector<int> male_shoes;
    for(int s=41; s<=46; s++) male_shoes.push_back(s); // 6 values

    while(male_heights.size() < n_male) male_heights.insert(male_heights.end(), male_heights.begin(), male_heights.end());
    while(male_shoes.size() < n_male) male_shoes.insert(male_shoes.end(), male_shoes.begin(), male_shoes.end());

    male_heights.resize(n_male);
    male_shoes.resize(n_male);

    shuffle_vector(male_heights);
    shuffle_vector(male_shoes);
    shuffle_vector(males);

    for(int i=0; i<n_male; i++){
        file << males[i % males.size()] << "," << male_heights[i] << "," << male_shoes[i] << ",1\n";
    }

    file.close();
    std::cout << "Distinct, non-overlapping dataset generated: distinct_dataset.csv\n";
    return 0;
}
