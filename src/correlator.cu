#include "correlator.cuh"
#include <cassert>

template <typename T>
__global__ void MultiTau::correlate<T>(T * new_values, size_t packet_size, T * shift_register, T * accumulator, T * insert_indexes, T * correlation){

};

template <typename T>
Correlator<T>::Correlator(size_t _num_bins, size_t _bin_size, size_t _num_sensors, bool _verbose){    
    num_bins = _num_bins;
    bin_size = _bin_size;
    num_sensors = _num_sensors;
    verbose = _verbose;    
};

template <typename T>
Correlator<T>::~Correlator(){
    free(correlation);
    free(taus);
};

template <typename T>
void Correlator<T>::alloc(){

};

template <typename T>
void Correlator<T>::correlate(T * new_values, size_t packet_size){

};

template <typename T>
void Correlator<T>::outputs(T * correlations, uint32_t* t){

};

template <typename T>
void Correlator<T>::reset(){

};