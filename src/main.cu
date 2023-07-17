#include <iostream>

#include "utils.hpp"
#include "options.hpp"
#include "correlator.cuh"


int main (int argc, char* argv[]){
    Options options(argc, argv);
    Correlator<int32_t> correlator(options.num_bins, options.bin_size, options.num_sensors, 8, 0, options.debug);

    std::cout << "Hello, World!" << std::endl ;

    return 0;
}