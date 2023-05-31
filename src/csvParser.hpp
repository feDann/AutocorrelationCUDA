#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


namespace utils {

    template <typename T>
    std::vector<T> parseCSVLine(const std::string &line){
        std::vector<T> result;
        std::stringstream lineStream(line);
        std::string cell;

        while(std::getline(lineStream, cell, ',')){
            result.push_back((T) stod(cell));
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