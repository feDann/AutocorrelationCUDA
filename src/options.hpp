#pragma once

#include <iostream>

#include <getopt.h>
#include <string>


#define DEBUG false
#define ITERATIONS 1
#define PARSE_FILE false
#define OUTPUT_FILE "out"
#define NUM_BINS 10
#define BIN_SIZE 32
#define NUM_SENSORS 1024


struct Options {
    bool debug = DEBUG;

    uint iterations = ITERATIONS;
    int packet_size=-1;
    size_t num_bins = NUM_BINS;
    size_t bin_size = BIN_SIZE;
    size_t num_sensors = NUM_SENSORS;

    bool results = false;
    bool parse_file = PARSE_FILE;

    std::string input_file;
    std::string output_file = OUTPUT_FILE;

    Options(int argc, char* argv[]){
        int opt;
        int option_index = 0;
        static struct option long_options[] = { {"debug", no_argument, 0, 'd'},
                                                {"results", no_argument, 0, 'r'},
                                                {"packets", required_argument, 0, 'p'},
                                                {"input_file", required_argument, 0, 'i'},
                                                {"output_file", optional_argument, 0, 'o'},
                                                {"iterations", optional_argument, 0, 'I'},
                                                {"sensors", optional_argument, 0, 's'},
                                                {"groups", optional_argument, 0, 'l'},
                                                {"group_size", optional_argument, 0, 'g'},
                                                {"help", no_argument, 0, 'h'},
                                                {0,0,0,0}};

        while(( opt = getopt_long(argc, argv, "hdrp:i:o:I:s:l:g:", long_options, &option_index) ) != EOF ) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'r':
                    results = true;
                    break;
                case 'p':
                    packet_size = atoi(optarg);
                    break;
                case 'i':
                    input_file = optarg;
                    parse_file = true;
                    break;
                case 'o':
                    output_file = optarg;
                    break;
                case 'I':
                    iterations = atoi(optarg);
                    break;
                case 's':
                    num_sensors = atoi(optarg);
                    break;
                case 'l':
                    num_bins = atoi(optarg);
                    break;
                case 'g':
                    bin_size = atoi(optarg);
                    break;
                case 'h':
                    std::cout << "This program is designed to perform autocorrelation calculations on sensor data using CUDA, a parallel computing platform."<< std::endl <<std::endl;
                    std::cout << "\t--debug, -d\t\t Activate debug prints" << std::endl;
                    std::cout << "\t--results, -r\t\t Prints to stdout the results of the autocorrelation" << std::endl;
                    std::cout << "\t--packets, -p\t\t Number of instant used per packets" << std::endl;
                    std::cout << "\t--input_file, -i\t\t Name of the input file containing the sensor data. If not given random data will e used for the calculation of the autocorrelation" << std::endl;
                    std::cout << "\t--output-file, -o \t\t Name of the output file. Correlation result will be saved into a csv file. If not given a default name will be used \"out.csv\"" << std::endl;
                    std::cout << "\t--iterations, -I\t\t Number of times that the calculation is repeated. If it is greater than one the calculation of the autocorrelation will be repeated multiple times on the same data" << std::endl;
                    std::cout << "\t--sensors, -s\t\t Number of Sensors present in the matrix of sensors" << std::endl;
                    std::cout << "\t--groups, -l\t\t Number of bins used for each correlator of the sensor matrix" << std::endl;
                    std::cout << "\t--group_size, -g\t\t Size of the bins for each correlator" << std::endl;
                    std::cout << "\t--help, -h\t\t Print this help message" << std::endl;
                    exit(0);
                    break;
                default:
                    break;
            }
        }

        if( packet_size == -1){
            std::cerr << "ERROR: --packets is a required argument. Use --help for more details" << std::endl;
            exit(1);
        }


    }
};