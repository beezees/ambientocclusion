C=g++ 
CFLAGS=-g -Wall -I. -I/usr/X11/include
LDFLAGS= -lstdc++ -L /usr/X11R6/lib/ -lglut -lGL -lGLU -lX11 -lXmu -lXi -lm -L /usr/lib
SOURCES=glm.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=glm

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	@rm -f $(EXECUTABLE) $(OBJECTS)
