.PHONY: clean md5sum wipe_output_data make default

CXX = g++
NVCC = nvcc

EXE = simplePIC3D
SRC = simplePIC3D.cu

INC = *.hpp

ARGS =  # Leave empty; arguments will be provided during execution

default: $(EXE)

$(EXE): $(SRC) $(INC)
	$(NVCC) -arch=sm_60 -o $@ $(SRC) -I ./include -L /usr/lib/x86_64-linux-gnu -L ./lib -lconfig -lfftw3 -lcurand -lcufft

run: $(EXE)
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib ./$(EXE) $(ARGS)

md5sum:
	md5sum out/global_quantities.txt

dat2xfm:
	cd ./out_h5 && sh dat2xdmf.sh && cd -

clean:
	rm -rf *.o $(EXE)

wipe_output_data:
	rm -rf out/* out_h5/fields*.h5 out_h5/fields*.xmf
