#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>


namespace utils {

    template <typename T> 
    T convertTo(const std::string &cell) {
        std::istringstream ss(cell);
        T value;

        ss >> value;

        if (ss.fail()){
            throw std::runtime_error("ERROR: Could not convert content to type");
        }

        return value;
    }

    template <typename T>
    std::vector<T> parseCSVLine(const std::string &line){
        std::vector<T> result;
        std::stringstream lineStream(line);
        std::string cell;

        while(std::getline(lineStream, cell, ',')){
            result.push_back(convertTo<T>(cell));
        }

        return result;
    }


    template <typename T>
    std::vector<T> parseCSV(const std::string &filepath) {

        std::vector<T> result;
        std::ifstream file(filepath);

        if(!file.is_open()) throw std::runtime_error("ERROR: Could not open file" + filepath);

        //Skip header line
        std::string line;
        std::getline(file, line);

        //Read data into result array
        while(std::getline(file, line)) {
            std::vector<T> lineData = parseCSVLine<T>(line);
            result.insert(result.end(), lineData.begin(), lineData.end());
        }

        file.close();
        return result;

    }

}