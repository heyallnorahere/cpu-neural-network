# cpu-neural-network

Simple neural network written in C#, .NET 7. Program uses a dataset from [here](http://yann.lecun.com/exdb/mnist/). Image file should be extracted as `data/images`, and label file should be extracted as `data/labels`.

## Usage
```
Basic usage:
    dotnet NeuralNetwork.dll [options] [data-file]

Options:
    --train
        Train the network
    
    --break-threshold <threshold>
        Stop training when the average absolute cost of a batch falls under <threshold>. Do not specify to train indefinitely.

Arguments:
    data-file:
        The JSON file to save/load weight and bias data from
```