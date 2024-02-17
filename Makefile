all:
	@mkdir -p bin
	g++ -o ./bin/streamer streamer.cpp -lavformat -lavcodec -lavutil
	g++ -o ./bin/receiver receiver.cpp -lavformat -lavcodec -lavutil
