PROJ = streamer

BIN_DIR = bin
all:
	@mkdir -p $(BIN_DIR)
	g++ -o $(BIN_DIR)/$(PROJ) streamer.cpp -lavformat -lavcodec -lavutil

run:
	./$(BIN_DIR)/$(PROJ)
