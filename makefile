CC = g++
INCLUDES = -I /opt/tools/eigen-eigen-ffa86ffb5570 
CFLAGS = -std=c++11 -O3 -DADEPT_STACK_THREAD_UNSAFE -ffast-math
CFLAGS_EVAL = -std=c++11 -O3
LIBS = -ladept
LIBS_EVAL = -fopenmp
SRCS = conv.cc utils.cc vecops.cc loss.cc
SRCS_EVAL = eval.cc
OUTPUT = conv.o
OUTPUT_EVAL = eval.o

conv:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS) -o $(OUTPUT) $(LIBS)
eval:
	$(CC) $(INCLUDES) $(CFLAGS_EVAL) $(SRCS_EVAL) -o $(OUTPUT_EVAL) $(LIBS_EVAL)
clean:
	$(RM) *.o *~
