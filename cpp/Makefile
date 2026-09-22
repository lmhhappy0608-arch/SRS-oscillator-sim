.PHONY: all clean run

all: build/generate_raw_grid

build/generate_raw_grid: main.cpp third_party/cnpy/cnpy.cpp third_party/cnpy/cnpy.h
	@mkdir -p build
	clang++ -std=c++17 -O3 -DNDEBUG -I third_party/cnpy \
		main.cpp third_party/cnpy/cnpy.cpp -lz -o build/generate_raw_grid

clean:
	rm -f build/generate_raw_grid

run: all
	./build/generate_raw_grid --help
