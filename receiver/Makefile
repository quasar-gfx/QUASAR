PROJ = receiver

CXX ?= g++

BIN_DIR = bin
SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.c)
OBJS = $(wildcard $(OBJ_DIR)/*.o)
OBJ_RULE = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
INCLUDES = -Iinclude
THIRD_PARTY = -Ithird_party

CXXFLAGS = -O2
LDFLAGS = -ldl -lglfw -lavformat -lavcodec -lavutil

all: setup build

setup:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(THIRD_PARTY)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(THIRD_PARTY)

$(BIN_DIR)/$(PROJ): $(OBJ_RULE)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

build: $(BIN_DIR)/$(PROJ)

clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/*

# g++ main.cpp glad.c -o OpenGLTest -ldl -lglfw -Iinclude

