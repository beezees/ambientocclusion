CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc -arch=sm_21 -ccbin  /usr/bin

# Common flags
COMMONFLAGS += $(INCLUDES)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_OPENGL := -lGL -lGLU

OBJS = coordinates.o
TARGET = coordinates 

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) 
	$(LINK) -o $(TARGET) $(OBJS) $(LIB_OPENGL)

clean:
	@rm -f *.o $(OBJS) $(TARGET) 
