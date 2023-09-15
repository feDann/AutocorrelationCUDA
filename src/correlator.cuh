#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define M 2 // Number of lag coalesced
namespace MultiTau {


    __inline__ __device__ int
    insert_until_bin(const int instants, const int num_bins);

    template <typename T>
    __global__ void 
    correlate ( T * new_values, 
                const int timepoints, 
                int instants_processed, 
                T * shift_register, 
                int * shift_positions, 
                T * accumulators, 
                T * correlation,  
                const int num_bins,
                const int num_sensors
                );

}

template<typename T>
class Correlator {

    public:
        Correlator(const int num_bins, const int bin_size, const int num_sensors, const int packet_size, const int device = 0, const bool debug = false);
        ~Correlator();

        void alloc();
        void correlate(const T * new_values, const int timepoints);
        void transfer();
        T get(const int sensor, const int lag);
        void reset();

    private:
        dim3 number_of_blocks;
        dim3 threads_per_block;

        int num_bins;
        int bin_size;
        int num_sensors;
        int num_sensors_per_block;
        int packet_size;

        int shared_memory_per_block;

        uint32_t max_tau;
        uint32_t num_taus;

        int instants_processed = 0;

        bool debug;
        bool transfered;

        T * correlation = nullptr;
        int * taus = nullptr;

        // Device variables
        T * d_shift_register = nullptr;
        int * d_shift_positions = nullptr;

        T * d_accumulators = nullptr;

        T * d_correlation = nullptr;
        T * d_new_values = nullptr;

};