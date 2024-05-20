GPU = -acc=gpu -D OPENACC__ -D NVPROF_
MULT = -acc=multicore -Mconcur=allcores
PGC = pgc++ -fast -O2

all: core mult gpu

core: task.cpp
	$(PGC) -o core task.cpp

mult: task.cpp
	$(PGC) $(MULT) -o mult task.cpp

gpu: task.cpp
	$(PGC) $(GPU) -o gpu task.cpp

clean:all
	rm gpu core mult