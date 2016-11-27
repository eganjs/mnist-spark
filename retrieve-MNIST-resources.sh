#!/bin/bash

file_urls=(
	"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
	"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
)

for file_url in ${file_urls[@]}; do
	wget $file_url --directory-prefix ./resources
done

gzip --decompress --force ./resources/*.gz

