using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public struct Layer
    {
        public Layer(int currentLayerSize, int previousLayerSize)
        {
            // nxa * axm = nxm
            Weights = new float[currentLayerSize, previousLayerSize];
            Biases = new float[currentLayerSize];
        }

        public float[,] Weights;
        public float[] Biases;

        [JsonIgnore]
        public bool Valid => Weights.GetLength(0) == Biases.Length;

        public void Step(Layer[] deltas, float eta)
        {
            if (!Valid)
            {
                throw new InvalidOperationException("Layer not valid!");
            }

            int currentSize = Weights.GetLength(0);
            int previousSize = Weights.GetLength(1);

            float scale = eta / deltas.Length;
            for (int i = 0; i < deltas.Length; i++)
            {
                var delta = deltas[i];
                for (int y = 0; y < currentSize; y++)
                {
                    Biases[y] -= delta.Biases[y] * scale;
                    for (int x = 0; x < previousSize; x++)
                    {
                        Weights[y, x] -= delta.Weights[y, x] * scale;
                    }
                }
            }
        }
    }

    public struct NetworkData
    {
        public int[] LayerSizes;
        public Layer[] Data;
    }

    public struct LabeledData
    {
        public float[] Input;
        public float[] ExpectedOutput;
    }

    public sealed class Network
    {
        public Network(IReadOnlyList<int> layerSizes)
        {
            mLayerSizes = layerSizes.ToArray();
            mLayers = new Layer[mLayerSizes.Length - 1];

            var random = new Random();
            for (int i = 0; i < mLayers.Length; i++)
            {
                int previousSize = layerSizes[i];
                int currentSize = layerSizes[i + 1];
                var layer = new Layer(currentSize, previousSize);

                for (int j = 0; j < currentSize; j++)
                {
                    layer.Biases[j] = random.NextSingle();
                    for (int k = 0; k < previousSize; k++)
                    {
                        layer.Weights[j, k] = (random.NextSingle() - 0.5f) * 2;
                    }
                }

                mLayers[i] = layer;
            }
        }

        public Network(NetworkData data)
        {
            int layerCount = data.LayerSizes.Length;
            int layerDataLength = data.Data.Length;

            if (layerCount != layerDataLength + 1)
            {
                throw new ArgumentException("Layer count mismatch!");
            }

            mLayerSizes = new int[layerCount];
            Array.Copy(data.LayerSizes, mLayerSizes, layerCount);

            mLayers = new Layer[layerDataLength];
            for (int i = 0; i < layerDataLength; i++)
            {
                var layer = new Layer(mLayerSizes[i + 1], mLayerSizes[i]);
                var layerData = data.Data[i];

                Array.Copy(layerData.Biases, layer.Biases, layer.Biases.Length);
                Array.Copy(layerData.Weights, layer.Weights, layer.Weights.Length);

                mLayers[i] = layer;
            }
        }

        public NetworkData Data => new NetworkData
        {
            LayerSizes = mLayerSizes,
            Data = mLayers
        };

        public int[] LayerSizes => mLayerSizes;
        public Layer[] Layers => mLayers;

        public void Evaluate(float[] input, out float[][] activations, out float[][] z)
        {
            z = new float[mLayers.Length][];
            activations = new float[mLayerSizes.Length][];
            activations[0] = input;

            for (int j = 0; j < mLayers.Length; j++)
            {
                var layer = mLayers[j];
                var previousActivations = activations[j];

                int layerSize = mLayerSizes[j + 1];
                var layerActivations = new float[layerSize];
                var layerZ = new float[layerSize];

                var composited = Program.Composite(layer.Weights, previousActivations);
                for (int k = 0; k < layerSize; k++)
                {
                    float neuronZ = layerZ[k] = composited[k] + layer.Biases[k];
                    layerActivations[k] = Program.Sigmoid(neuronZ);
                }

                activations[j + 1] = layerActivations;
                z[j] = layerZ;
            }
        }

        public float[] Evaluate(float[] input)
        {
            Evaluate(input, out float[][] activations, out _);
            return activations[^1];
        }

        public void Backpropagate(float[][] activations, float[][] z, float[] expected, out Layer[] deltas)
        {
            deltas = new Layer[mLayers.Length];

            var delta = new float[mLayerSizes[^1]];
            for (int i = 0; i < delta.Length; i++)
            {
                float activation = activations[^1][i];
                float zValue = z[^1][i];
                delta[i] = Program.CostDerivative(activation, expected[i]) * Program.SigmoidPrime(zValue);
            }

            deltas[^1] = new Layer
            {
                Biases = delta,
                Weights = Program.Composite(delta, Program.Transpose(activations[^2]))
            };

            for (int i = 2; i <= deltas.Length; i++)
            {
                delta = Program.Composite(Program.Transpose(mLayers[^(i-1)].Weights), delta);
                for (int j = 0; j < delta.Length; j++)
                {
                    delta[j] *= Program.SigmoidPrime(z[^i][j]);
                }

                deltas[^i] = new Layer
                {
                    Biases = delta,
                    Weights = Program.Composite(delta, Program.Transpose(activations[^(i+1)]))
                };
            }
        }

        public void TrainOnBatch(LabeledData[] batch, float eta, out float averageAbsoluteCost)
        {
            int batchSize = batch.Length;
            var deltas = new Layer[mLayers.Length, batchSize];

            averageAbsoluteCost = 0f;
            for (int i = 0; i < batchSize; i++)
            {
                var pass = batch[i];
                Evaluate(pass.Input, out float[][] activations, out float[][] z);
                
                for (int j = 0; j < mLayerSizes[^1]; j++)
                {
                    float cost = Program.Cost(activations[^1][j], pass.ExpectedOutput[j]);
                    averageAbsoluteCost += MathF.Abs(cost);
                }

                Backpropagate(activations, z, pass.ExpectedOutput, out Layer[] passDeltas);
                for (int j = 0; j < passDeltas.Length; j++)
                {
                    deltas[j, i] = passDeltas[j];
                }
            }

            for (int i = 0; i < mLayers.Length; i++)
            {
                var layerDeltas = new Layer[batchSize];
                for (int j = 0; j < batchSize; j++)
                {
                    layerDeltas[j] = deltas[i, j];
                }

                mLayers[i].Step(layerDeltas, eta);
            }

            averageAbsoluteCost /= batchSize * mLayerSizes[^1];
        }

        private readonly int[] mLayerSizes;
        private readonly Layer[] mLayers;
    }
}