CXX=nvcc

CUDAFLAGS=--ptxas-options=-v -O4 -m64 -arch compute_61 -code sm_61 -Xptxas -dlcm=ca -Xcompiler -D_FORCE_INLINES -lineinfo

BIN_FOLDER=bin
SRC_FOLDER=src

FILES=${SRC_FOLDER}/main.cu ${SRC_FOLDER}/utils.cpp ${SRC_FOLDER}/correlator.cu

all: main

clean:
	rm $(BIN_FOLDER)/*

main:
	mkdir -p $(BIN_FOLDER)
	$(CXX) $(FILES) $(CUDAFLAGS) -o $(BIN_FOLDER)/main

debug:
	mkdir -p $(BIN_FOLDER)
	$(CXX) $(FILES) $(CUDAFLAGS) -D_DEBUG_MODE -o $(BIN_FOLDER)/main 

