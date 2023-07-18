#include <iostream>

#include "utils.hpp"
#include "options.hpp"
#include "correlator.cuh"


int main (int argc, char* argv[]){
    Options options(argc, argv);
    Correlator<int32_t> correlator(options.num_bins, options.bin_size, options.num_sensors, 8, 0, options.debug);
    int32_t data[] = {1,2,3,4,5,6,7,8,9,0};

    correlator.alloc();
    correlator.correlate(data, 1);
    correlator.reset();

    return 0;
}