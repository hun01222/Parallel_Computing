# Makefile for building main.cpp (ONNX → TensorRT C++ demo)

CXX      := g++
CXXFLAGS := -std=c++17 -O3 \
            -I/usr/include/aarch64-linux-gnu \
            -I/usr/include/x86_64-linux-gnu \
            -I/usr/local/cuda/include \
            -Wno-deprecated-declarations

LDFLAGS  := -L/usr/lib/aarch64-linux-gnu \
            -L/usr/local/cuda/lib64 \
            -lnvinfer \
            -lnvonnxparser \
            -lcudart

SRCS     := main.cpp
OBJS     := $(SRCS:.cpp=.o)
TARGET   := test_resnet101

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
