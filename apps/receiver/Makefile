PROJ = receiver

CXX ?= g++

BIN_DIR = bin
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include
COMMON_DIR = ../common

SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.c) $(wildcard $(COMMON_DIR)/src/*.cpp)
OBJS = $(wildcard $(OBJ_DIR)/*.o)
OBJ_RULE = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(SRCS:$(COMMON_DIR)/src/%.cpp=$(OBJ_DIR)/%.o)
INCLUDES = -I$(INCLUDE_DIR) -I$(COMMON_DIR)/include
THIRD_PARTY = -I../third_party

CXXFLAGS = -O2 -std=c++17
LDFLAGS = -ldl -lglfw -lavformat -lavcodec -lavutil -lswscale

all: setup build

setup:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(THIRD_PARTY)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(THIRD_PARTY)

$(OBJ_DIR)/%.o: $(COMMON_DIR)/src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(THIRD_PARTY)

$(BIN_DIR)/$(PROJ): $(OBJ_RULE)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

build: $(BIN_DIR)/$(PROJ)

run: $(BIN_DIR)/$(PROJ)
	@echo "Running $(PROJ)"
	@./$(BIN_DIR)/$(PROJ)

clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
