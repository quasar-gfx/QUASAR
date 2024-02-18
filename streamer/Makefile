PROJ = streamer

CXX ?= g++

BIN_DIR = bin

all:
	@mkdir -p $(BIN_DIR)
	$(CXX) -std=c++17 -o $(BIN_DIR)/$(PROJ) streamer.cpp -lavformat -lavcodec -lavutil

run: $(BIN_DIR)/$(PROJ)
	@echo "Running $(PROJ)"
	@./$(BIN_DIR)/$(PROJ)
