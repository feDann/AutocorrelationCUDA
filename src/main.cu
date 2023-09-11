#include <iostream>
#include <vector>
#include <chrono>

#include "utils.hpp"
#include "options.hpp"
#include "correlator.cuh"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

int main (int argc, char* argv[]){
    Options options(argc, argv);
    Correlator<uint32_t> correlator(options.num_bins, options.bin_size, options.num_sensors, options.num_sensors_per_block, 0, options.debug);
    
    if (options.debug) std::cout << "Reading input file" << std::endl;
    std::vector<uint32_t> data = utils::parseCSV<uint32_t>(options.input_file);
    
    correlator.alloc();

    size_t total_instants = data.size() / options.num_sensors;
    size_t total_packets = total_instants / options.packet_size;

    for(size_t i = 0; i < options.iterations; ++i){
        correlator.reset();

        auto start = clock_type::now();

        if (options.debug) std::cout << "Starting iteration: " << i << std::endl;

        for (size_t j = 0; j < total_packets; ++j){
            if (options.debug) std::cout << "Executing packet " << j << std::endl;

            size_t starting_position = j * options.packet_size * options.num_sensors;
            correlator.correlate(data.data() + starting_position, options.packet_size);
        }

        auto end = clock_type::now();
        auto duration = chrono::duration<double>(end-start);

        if (!options.debug) std::cout << duration.count() << std::endl;

    }

    correlator.transfer();

    if (options.results) {
        std::ofstream outputFile(options.output_file);
		auto taus = utils::generateTaus(total_instants, options.bin_size, options.num_bins);

		for (int lag = 0; lag < taus.size(); lag++){
			outputFile << taus[lag];
			for (int sensor = 0; sensor < options.num_sensors; ++sensor){
				auto value = correlator.get(sensor, lag);
				outputFile << ',' << value;
			}
			outputFile << std::endl;
		}

		outputFile.close();
    }


    return 0;
}