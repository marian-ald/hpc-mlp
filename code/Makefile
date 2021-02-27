
# Compiling options
CFLAGS=-Wall -std=c++11

# Compiler
CC = g++ -fopenmp
CC_ser = g++

# Executables
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXEC = mlp


FLAGS = -DVERBOSE=0
all : $(EXEC)

# Create binary files
mlp: utils.o main.o MLP.o
	$(CC) $(CFLAGS) -o $@ $+

# Objects
%.o: %.cpp
	$(CC) $(FLAGS) $(CFLAGS) -c $<

# Clear intermediary object files
clean:
	rm -f *.o *~

# Remove executable
clean_all: clean
	rm $(EXEC)

run: mlp
	./mlp