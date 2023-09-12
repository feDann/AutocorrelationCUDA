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
    Correlator<uint32_t> correlator(options.num_bins, options.bin_size * 2, options.num_sensors, 0, options.debug);
    
    if (options.debug) std::cout << "[INFO] Reading input file" << std::endl;
    std::vector<uint32_t> data = utils::parse_csv<uint32_t>(options.input_file);
    
    correlator.alloc();

    size_t total_instants = data.size() / options.num_sensors;
    size_t total_packets = total_instants / options.packet_size;

    for(size_t i = 0; i < options.iterations; ++i){
        correlator.reset();

        auto start = clock_type::now();

        if (options.debug) std::cout << "[INFO] Starting iteration: " << i << std::endl;

        for (size_t j = 0; j < total_packets; ++j){
            if (options.debug) std::cout << "[INFO] Executing packet " << j << std::endl;

            size_t starting_position = j * options.packet_size * options.num_sensors;
            correlator.correlate(data.data() + starting_position, options.packet_size);
        }

        auto end = clock_type::now();
        auto duration = chrono::duration<double>(end-start);

        if (!options.debug) std::cout << duration.count() << std::endl;

    }

    correlator.transfer();

    if (options.results) {
        std::ofstream output_file(options.output_file);
		auto taus = utils::generate_taus(total_instants, options.bin_size, options.num_bins);

		for (int lag = 0; lag < taus.size(); lag++){
			output_file << taus[lag];
			for (int sensor = 0; sensor < options.num_sensors; ++sensor){
				auto value = correlator.get(sensor, lag);
				output_file << ',' << value;
			}
			output_file << std::endl;
		}

        // for (size_t bin = 0 ; bin < options.num_bins; ++bin) {
            // for(size_t channel = 0; channel < options.bin_size ; ++channel) {
                // output_file << "," << correlator.correlation[bin * 8 * options.bin_size + channel] ;
            // }
            // output_file << std::endl;
        // }

		output_file.close();
    }


    return 0;
}