CC = g++
BIN = main
OBJS = $(BUILD)/$(BIN).o $(BUILD)/filters.o $(BUILD)/classifier.o
CFLAGS = -std=c++17
INC = -I./include
LIB = `pkg-config --libs opencv`

BUILD = ./src/obj
SRC = ./src

$(BIN): $(OBJS)
	$(CC) -o $(BIN) $(OBJS) $(INC) $(LIB) $(LIBS)

$(BUILD)/%.o: $(SRC)/%.cpp 
	$(CC) -c -o $@ $< $(INC) $(CFLAGS)

clean: 
	rm $(BUILD)/*.o
	rm $(BIN)