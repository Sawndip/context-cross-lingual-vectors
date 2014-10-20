CC = g++
INCLUDES = -I /opt/tools/eigen-eigen-ffa86ffb5570 
CFLAGS = -std=c++11 -O3 -DADEPT_STACK_THREAD_UNSAFE -ffast-math
LIBS = -ladept
SRCS = train.cc utils.cc vecops.cc
SRCS_CONV = conv.cc utils.cc vecops.cc
SRCS_MAN = train-manual.cc utils.cc vecops.cc
OUTPUT = train.o
OUTPUT_CONV = conv.o
OUTPUT_MAN = train-manual.o

compile:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS) -o $(OUTPUT) $(LIBS)
conv:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS_CONV) -o $(OUTPUT_CONV) $(LIBS)
clean:
	$(RM) *.o *~
manual:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS_MAN) -o $(OUTPUT_MAN) $(LIBS)
