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
        Correlator(size_t num_bins, size_t bin_size, size_t num_sensors, size_t num_sensors_per_block = 8, int device = 0, bool verbose = false);
        ~Correlator();

        void alloc();
        void correlate(T * new_values, size_t packet_size);
        void outputs(T * correlations, uint32_t* t);
        T get(size_t sensor, size_t lag);
        void reset();

    private:
        dim3 number_of_blocks;
        dim3 threads_per_block;

        size_t num_bins;
        size_t bin_size;
        size_t num_sensors;
        size_t num_sensors_per_block;

        uint32_t max_tau;

        bool verbose;

        T * correlation;
        uint32_t * taus;

        // device variables
        T * d_shift_register;
        T * d_accumulator;
        T * d_correlation;
        int * d_insert_indexes;
        T * d_new_values;
};