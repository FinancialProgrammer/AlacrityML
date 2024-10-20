COM = g++
CU_COM = nvcc

LIBS = -lAlacrity

FLAGS = ${LIBS} -O3 # -Ofast
CU_FLAGS = ${LIBS} -O3 --extended-lambda
CU_COM_FLAGS = -x cu ${CU_FLAGS}

all: setup.out stdalgo.out extalgo.out math.out linalgo.out random.out
cu: setup.cu.out stdalgo.cu.out extalgo.cu.out math.cu.out linalgo.cu.out random.cu.out

# Unit Tests
%.out: %.o
	${COM} $^ -o $@ ${FLAGS}

%.cu.out: %.cu.o
	${CU_COM} $^ -o $@ ${CU_FLAGS}

# END

%.o: %.cpp
	${COM} -c $^ -o $@ ${FLAGS}

%.cu.o: %.cpp
	${CU_COM} -c $^ -o $@ ${CU_COM_FLAGS}

clean:
	rm *.out
