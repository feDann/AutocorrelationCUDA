#include "correlator.cuh"
#include <cassert>
#include <cmath>

#ifdef _DEBUG_BUILD
#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ <<": ERROR: " << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ <<": ERROR: " << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#else
#define CHECK(call) call;                                                                   
#define CHECK_KERNELCALL()                                                         
#endif // _DEBUG_BUILD


// Macros

#define SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block) \
        (sensor) * (bin_size) + (bin) * (bin_size) * (num_sensors_per_block) + (channel)

#define SHARED_OFF_B(sensor, bin, num_sensors_per_block) \
        (sensor) + (bin) * (num_sensors_per_block)

#define GLOBAL_OFF(sensor, bin, channel, bin_size, num_sensors_per_block, num_bins, first_block_sensor) \
        (first_block_sensor) * (num_bins) * (bin_size) + SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)

#define GLOBAL_OFF_B(sensor, bin, num_sensors_per_block, num_bins, first_block_sensor) \
        (first_block_sensor) * (num_bins) + SHARED_OFF_B(sensor, bin, num_sensors_per_block)


// Kernels
template <typename T>
__global__ void 
MultiTau::correlate<T>(T * new_values, const size_t timepoints, size_t instants_processed, T * shift_register, int * shift_positions, T * accumulators, int * num_accumulators, T * correlation, const size_t num_bins){

    size_t num_sensors_per_block = blockDim.y;
    size_t bin_size = blockDim.x;
    size_t num_sensors = gridDim.x * num_sensors_per_block;

    size_t sensor = threadIdx.y;  // relative sensor id inside the block -> 0..num_sensors_per_block
    size_t channel = threadIdx.x; // channels goes from 0..bin_size

    size_t first_block_sensor = blockIdx.x * num_sensors_per_block; // id for the first sensor of the block
    size_t sensor_gp = first_block_sensor + sensor; // gp stands for global-position, this is the global sensor id

    // Due to templates memory needs to be assigned as unsigned char 
    extern __shared__  unsigned char total_shared_memory[];

    int * block_shift_pos = reinterpret_cast<int *>(&total_shared_memory);
    int * block_num_accumulators = &block_shift_pos[num_sensors_per_block * num_bins];
    T * block_shift = reinterpret_cast<T *>(&block_num_accumulators[num_sensors_per_block * num_bins]);
    T * block_correlation = &block_shift[num_sensors_per_block * num_bins * bin_size];
    T * block_accumulators = &block_correlation[num_sensors_per_block * num_bins * bin_size];


    // Copy correlator arrays from global memory to shared memory
    for (size_t bin = 0; bin < num_bins; ++bin) {
        block_shift[SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)] =  shift_register[GLOBAL_OFF(sensor, bin, channel, bin_size, num_sensors_per_block, num_bins, first_block_sensor)];
        block_correlation[SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)] =  correlation[GLOBAL_OFF(sensor, bin, channel, bin_size, num_sensors_per_block, num_bins, first_block_sensor)];
    }

    if (channel < num_bins) {  // full threads are not required anymore, for each sensor we need to copy only num_bins accumulator, num_accumulator and shift_positions; the name channel migh be misleading here, consider it as "bin" for the next lines 
        block_num_accumulators[SHARED_OFF_B(sensor, channel, num_sensors_per_block)] = num_accumulators[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)];
        block_accumulators[SHARED_OFF_B(sensor, channel, num_sensors_per_block)] = accumulators[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)];
        block_shift_pos[SHARED_OFF_B(sensor, channel, num_sensors_per_block)] = shift_positions[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)];
    }  

    __syncthreads();

    // Add new point of the series to the correlator
    for (unsigned int instant = 0; instant < timepoints ; ++instant) {

        int insert_channel_fb = block_shift_pos[SHARED_OFF_B(sensor, 0, num_sensors_per_block)]; // fb stands for first bin
        T new_value = new_values[instant * num_sensors + sensor_gp];
        
        if (channel == 0) { // only one thread add the new_value to the sensor shift register
            block_shift[SHARED_OFF(sensor, 0, insert_channel_fb, bin_size, num_sensors_per_block)] = new_value;
            block_accumulators[SHARED_OFF_B(sensor, 0, num_sensors_per_block)] += new_value;
            block_num_accumulators[SHARED_OFF_B(sensor, 0, num_sensors_per_block)] += 1;
        }

        __syncthreads();

        block_correlation[SHARED_OFF(sensor, 0, channel, bin_size, num_sensors_per_block)] +=  block_shift[SHARED_OFF(sensor, 0, insert_channel_fb, bin_size, num_sensors_per_block)] * block_shift[SHARED_OFF(sensor, 0, (insert_channel_fb - channel + bin_size) & ( bin_size - 1 ), bin_size, num_sensors_per_block)];
        block_shift_pos[SHARED_OFF_B(sensor, 0, num_sensors_per_block)] = (insert_channel_fb + 1) & (bin_size-1);
        
        __syncthreads();
        
        for(unsigned int bin = 1; bin < num_bins && block_num_accumulators[SHARED_OFF_B(sensor, bin - 1, num_sensors_per_block)] == M  ; ++bin) {
            __syncthreads();

            int insert_channel = block_shift_pos[SHARED_OFF_B(sensor, bin, num_sensors_per_block)];
            
            if (channel == 0) { // only one thread add the new_value to the sensor shift register
                block_shift[SHARED_OFF(sensor, bin, insert_channel, bin_size, num_sensors_per_block)] = block_accumulators[SHARED_OFF_B(sensor, bin-1, num_sensors_per_block)];
                block_accumulators[SHARED_OFF_B(sensor, bin, num_sensors_per_block)] += block_accumulators[SHARED_OFF_B(sensor, bin-1, num_sensors_per_block)];
                block_num_accumulators[SHARED_OFF_B(sensor, bin, num_sensors_per_block)] += 1;
            }
            __syncthreads();

            block_accumulators[SHARED_OFF_B(sensor, bin-1, num_sensors_per_block)] = 0;
            block_num_accumulators[SHARED_OFF_B(sensor, bin-1, num_sensors_per_block)] = 0;

            if (channel >= bin_size/M) {
                block_correlation[SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)] +=  block_shift[SHARED_OFF(sensor, bin, insert_channel, bin_size, num_sensors_per_block)] * block_shift[SHARED_OFF(sensor, bin, (insert_channel - channel + bin_size) & (bin_size -1), bin_size, num_sensors_per_block)];
            }
            block_shift_pos[SHARED_OFF_B(sensor, bin, num_sensors_per_block)] = (insert_channel + 1) & (bin_size-1);

        }

        __syncthreads();

    }
    
    // Copy correlator arrays from global memory to shared memory
    for (size_t bin = 0; bin < num_bins; ++bin) {
        shift_register[GLOBAL_OFF(sensor, bin, channel, bin_size, num_sensors_per_block, num_bins, first_block_sensor)] = block_shift[SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)];
        correlation[GLOBAL_OFF(sensor, bin, channel, bin_size, num_sensors_per_block, num_bins, first_block_sensor)] = block_correlation[SHARED_OFF(sensor, bin, channel, bin_size, num_sensors_per_block)];
    }

    if (channel < num_bins) {  // full threads are not required anymore, for each sensor we need to copy only num_bins accumulator, num_accumulator and shift_positions; the name channel migh be misleading here, consider it as "bin" for the next lines 
        num_accumulators[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)] = block_num_accumulators[SHARED_OFF_B(sensor, channel, num_sensors_per_block)];
        accumulators[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)] = block_accumulators[SHARED_OFF_B(sensor, channel, num_sensors_per_block)];
        shift_positions[GLOBAL_OFF_B(sensor, channel, num_sensors_per_block, num_bins, first_block_sensor)] = block_shift_pos[SHARED_OFF_B(sensor, channel, num_sensors_per_block)];
    }

};


template <typename T>
Correlator<T>::Correlator(size_t t_num_bins, size_t t_bin_size, size_t t_num_sensors, int t_device, bool t_debug){    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, t_device);
    
    num_bins = t_num_bins;
    bin_size = t_bin_size;
    num_sensors = t_num_sensors;
    debug = t_debug;

    max_tau = bin_size * std::pow(2, num_bins);
    num_taus = bin_size * num_bins;

    //                        accumulators    shift_registers and outputs          accumulator and num_accumulator 
    shared_memory_per_block = (num_bins + 2 * (num_bins * bin_size) ) * sizeof(T) + 2 * (num_bins) * sizeof(int);
    num_sensors_per_block = std::floor((double) device_properties.sharedMemPerBlock / shared_memory_per_block);

    shared_memory_per_block *= num_sensors_per_block;

    assert(shared_memory_per_block <= device_properties.sharedMemPerBlock && num_sensors_per_block > 0 && "ERROR: current configuration exceed device shared memory limits");

    if (num_sensors_per_block <= MIN_SENSORS_PER_BOOCK && debug) {
        std::cout << "[WARNING] num_sensors_per_block is low, consider changing the current configuration of bin_size and num_bins" << std::endl;
    }

    number_of_blocks = dim3(num_sensors / num_sensors_per_block, 1 , 1);
    threads_per_block = dim3(bin_size, num_sensors_per_block, 1);

    if (debug){
        std::cout << "[INFO] Number of bins: " << num_bins << std::endl;
        std::cout << "[INFO] Size of bins: " << bin_size << std::endl;
        std::cout << "[INFO] Number of sensors: " << num_sensors << std::endl;
        std::cout << "[INFO] Number of sensors per block: " << num_sensors_per_block << std::endl;
        std::cout << "[INFO] Max tau possible: " << max_tau << std::endl;
        std::cout << "[INFO] Number of taus possible: " << num_taus << std::endl;
        std::cout << "[INFO] Shared Memory per block: " << shared_memory_per_block << " B" << std::endl;
        std::cout << "[INFO] Maximum shared memory available: " << device_properties.sharedMemPerBlock << " B" << std::endl;
        std::cout << "[INFO] Number of blocks: (" << number_of_blocks.x << "," << number_of_blocks.y << "," << number_of_blocks.z << ")" << std::endl;
        std::cout << "[INFO] Threads per blocks: (" << threads_per_block.x << "," << threads_per_block.y << "," << threads_per_block.z << ")" << std::endl;
    }    
};

template <typename T>
Correlator<T>::~Correlator(){
    // Free host memory
    if (correlation != nullptr){
        free(correlation);
    }
    if (taus != nullptr){
        free(taus);
    }

    // Free device memory
    if (d_shift_register != nullptr){
        CHECK(cudaFree(d_shift_register));
    }

    if (d_shift_positions != nullptr){
        CHECK(cudaFree(d_shift_positions));
    }

    if (d_accumulators != nullptr){
        CHECK(cudaFree(d_accumulators));
    }

    if (d_num_accumulators != nullptr){
        CHECK(cudaFree(d_num_accumulators));
    }

    if (d_correlation != nullptr){
        CHECK(cudaFree(d_correlation));
    }

    cudaDeviceReset();
};

template <typename T>
void Correlator<T>::alloc(){
    if (debug) std::cout << "[INFO] Allocating device arrays into global memory" << std::endl;

    CHECK(cudaMalloc(&d_shift_register, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_shift_positions, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMalloc(&d_accumulators, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMalloc(&d_num_accumulators, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMalloc(&d_correlation, num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "[INFO] Initializing device arrays" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_shift_positions, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_accumulators, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_num_accumulators, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "[INFO] Alocating device arrays" << std::endl;

    correlation = (T*)malloc(num_taus * num_sensors * sizeof(T));

};

template <typename T>
void Correlator<T>::correlate(T * new_values, size_t timepoints){

    transfered = false;

    if (debug) std::cout << "[INFO] Allocating and copying new values to gpu array" << std::endl;

    CHECK(cudaMalloc(&d_new_values, timepoints * num_sensors * sizeof(T)));
    CHECK(cudaMemcpy(d_new_values, new_values, timepoints * num_sensors * sizeof(T), cudaMemcpyHostToDevice));

    if (debug) std::cout << "[INFO] Starting correlation" << std::endl;

    MultiTau::correlate<T><<<number_of_blocks, threads_per_block, shared_memory_per_block>>>(d_new_values, timepoints, instants_processed, d_shift_register, d_shift_positions, d_accumulators, d_num_accumulators, d_correlation, num_bins);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
    
    if (d_new_values != nullptr){
        cudaFree(d_new_values);
    }

    if (debug) std::cout << "[INFO] Instant Processed: " << instants_processed << std::endl;

    instants_processed += timepoints;
};

template <typename T>
void Correlator<T>::transfer(){    

    if (debug) std::cout << "[INFO] Transfering data from device memory to host memory" << std::endl;

    if (!transfered){
        CHECK(cudaMemcpy(correlation, d_correlation, num_taus * num_sensors * sizeof(T), cudaMemcpyDeviceToHost));
        transfered = true;
    }

    if (debug) std::cout << "[INFO] Data transfered" << std::endl;
}


template <typename T>
T Correlator<T>::get(size_t sensor, size_t lag){
    assert(transfered && "ERROR: Data not transfered from device memory to host memory");
    
    int block = std::floor((double) sensor / num_sensors_per_block);
    int sensor_rp = sensor - block * num_sensors_per_block;
    
    if (lag < bin_size)
        return correlation[block * num_sensors_per_block * num_bins * bin_size + sensor_rp * bin_size + lag];
    
    int bin = std::ceil((double)(lag - bin_size + 1) / (double)(bin_size/2));
    int channel = (lag - bin_size) - (bin_size/2) * (bin-1) + (bin_size/2);

    return correlation[block * num_sensors_per_block * num_bins * bin_size + sensor_rp * bin_size  +  bin * num_sensors_per_block * bin_size + channel];
};

template <typename T>
void Correlator<T>::reset(){

    if (debug) std::cout << "[INFO] Resetting all device arrays to zero" << std::endl;

    CHECK(cudaMemset(d_shift_register, 0, num_bins * bin_size * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_shift_positions, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_accumulators, 0, num_bins * num_sensors * sizeof(T)));
    CHECK(cudaMemset(d_num_accumulators, 0, num_bins * num_sensors * sizeof(int)));

    CHECK(cudaMemset(d_correlation, 0 , num_taus * num_sensors * sizeof(T)));

    if (debug) std::cout << "[INFO] Resetting all host arrays to zero" << std::endl;

    memset(correlation, 0, num_taus * num_sensors * sizeof(T));

    instants_processed = 0;
    transfered = false;
};

// Needed for the template
template class Correlator<int8_t>;
template class Correlator<int16_t>;
template class Correlator<int32_t>;
template class Correlator<int64_t>;

template class Correlator<uint8_t>;
template class Correlator<uint16_t>;
template class Correlator<uint32_t>;
template class Correlator<uint64_t>;

template class Correlator<double>;
template class Correlator<float>;