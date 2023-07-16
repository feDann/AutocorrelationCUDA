#pragma once

#include <cuda_runtime.h>
#include <iostream>
namespace MultiTau {

    template <typename T>
    __global__ void correlate (T * new_values, size_t packet_size, T * shift_register, T * accumulator, T * insert_indexes, T * correlation);

}

template<typename T>
class Correlator {

    public:
        Correlator(size_t num_bins, size_t bin_size, size_t num_sensors, bool verbose = false);
        ~Correlator();

        void alloc();
        void correlate(T * new_values, size_t packet_size);
        void outputs(T * correlations, uint32_t* t);
        void reset();

    private:
        size_t num_bins;
        size_t bin_size;
        size_t num_sensors;

        bool verbose;

        T * correlation;
        uint32_t * taus;

        // device variables
        T * d_shift_register;
        T * d_accumulator;
        T * d_insert_indexes;
        T * d_correlation;
        T * d_new_values;


};