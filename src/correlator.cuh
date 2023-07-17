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
        Correlator(size_t num_bins, size_t bin_size, size_t num_sensors, size_t num_sensors_per_block = 8, int device = 0, bool debug = false);
        ~Correlator();

        void alloc();
        void correlate(T * new_values, size_t timepoints);
        T get(size_t sensor, size_t lag);
        void reset();

    private:
        dim3 number_of_blocks;
        dim3 threads_per_block;

        size_t num_bins;
        size_t bin_size;
        size_t num_sensors;
        size_t num_sensors_per_block;

        size_t shared_memory_per_block;

        uint32_t max_tau;
        uint32_t num_taus;

        bool debug;

        T * correlation = NULL;
        uint32_t * taus = NULL;

        // device variables
        T * d_shift_register = NULL;
        T * d_accumulator = NULL;
        T * d_correlation = NULL;
        int * d_insert_indexes = NULL;
        T * d_new_values = NULL;
};