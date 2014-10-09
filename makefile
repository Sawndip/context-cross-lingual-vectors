CC = g++
INCLUDES = -I /opt/tools/eigen-eigen-ffa86ffb5570 
CFLAGS = -std=c++11
LIBS = -ladept
SRCS = train.cc utils.cc vecops.cc
SRCS_TEST = test.cc utils.cc vecops.cc
SRCS_MAN = train-manual.cc utils.cc vecops.cc
OUTPUT = train.o
OUTPUT_TEST = test.o
OUTPUT_MAN = train-manual.o

compile:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS) -o $(OUTPUT) $(LIBS)
clean:
	$(RM) *.o *~
manual:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS_MAN) -o $(OUTPUT_MAN) $(LIBS)
test:
	$(CC) $(INCLUDES) $(CFLAGS) $(SRCS_TEST) -o $(OUTPUT_TEST) $(LIBS)
