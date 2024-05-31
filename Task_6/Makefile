GPU = -acc=gpu -D NVPROF_
MULT = -acc=multicore 
CORE = -acc=host
ADD = -lboost_program_options
PGC = pgc++ -fast -O2

all: core mult gpu

core: task.cpp
	$(PGC) $(CORE) $(ADD) -o core task.cpp

mult: task.cpp
	$(PGC) $(MULT) $(ADD) -o mult task.cpp

gpu: task.cpp
	$(PGC) $(GPU) $(ADD) -o gpu task.cpp

clean:all
	rm gpu core mult