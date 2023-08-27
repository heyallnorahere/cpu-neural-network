using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNetwork
{
    internal struct ValueConfidence
    {
        public int Value;
        public float Confidence;
    }

    internal struct ImageResult
    {
        public int ImageNumber;
        public int Value;
        public ValueConfidence[] Confidence;
        public double TimeElapsed;
    }

    internal struct ResultTable
    {
        public List<ImageResult> ImageResults;
        public int[] LayerSizes;
    }

    internal static class Program
    {
        private static readonly Random sRNG;
        static Program()
        {
            sRNG = new Random();
        }

        public static void Shuffle<T>(this T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int i = sRNG.Next(n--);
                (array[i], array[n]) = (array[n], array[i]);
            }
        }

        private static int ReadInt32WithEndianness(this BinaryReader reader)
        {
            var bytes = reader.ReadBytes(Marshal.SizeOf<int>());
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            return BitConverter.ToInt32(bytes);
        }

        // http://yann.lecun.com/exdb/mnist/
        public static float[,,] ReadImageData(Stream stream)
        {
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            if (reader.ReadInt32WithEndianness() != 0x803)
            {
                throw new IOException("Invalid magic number!");
            }

            int imageCount = reader.ReadInt32WithEndianness();
            int rowCount = reader.ReadInt32WithEndianness();
            int columnCount = reader.ReadInt32WithEndianness();

            var result = new float[imageCount, rowCount, columnCount];
            for (int i = 0; i < imageCount; i++)
            {
                for (int y = 0; y < rowCount; y++)
                {
                    for (int x = 0; x < columnCount; x++)
                    {
                        byte value = reader.ReadByte();
                        result[i, x, y] = (float)value / byte.MaxValue;
                    }
                }
            }

            return result;
        }

        public static int[] ReadLabelData(Stream stream)
        {
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            if (reader.ReadInt32WithEndianness() != 0x801)
            {
                throw new IOException("Invalid magic number!");
            }

            int labelCount = reader.ReadInt32WithEndianness();
            var result = new int[labelCount];
            for (int i = 0; i < labelCount; i++)
            {
                result[i] = reader.ReadByte();
            }

            return result;
        }

        public static float[] GetInputData(float[,,] imageData, int index)
        {
            int width = imageData.GetLength(1);
            int height = imageData.GetLength(2);

            var inputData = new float[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = y * width + x;
                    inputData[i] = imageData[index, x, y];
                }
            }

            return inputData;
        }

        public static void Main(string[] args)
        {
            bool train = false;
            float breakThreshold = -1f;

            var miscArgs = new List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                string argument = args[i];
                switch (argument)
                {
                    case "--train":
                        train = true;
                        break;
                    case "--break-threshold":
                        breakThreshold = float.Parse(args[++i]);
                        break;
                    default:
                        miscArgs.Add(argument);
                        break;
                }
            }

            var serializer = JsonSerializer.Create(new JsonSerializerSettings
            {
                ContractResolver = new DefaultContractResolver
                {
                    NamingStrategy = new SnakeCaseNamingStrategy()
                },
                Formatting = Formatting.Indented
            });

            float[,,] imageData;
            using (var stream = new FileStream("data/images", FileMode.Open, FileAccess.Read))
            {
                imageData = ReadImageData(stream);
            }

            int[] labelData;
            using (var stream = new FileStream("data/labels", FileMode.Open, FileAccess.Read))
            {
                labelData = ReadLabelData(stream);
            }

            int count = imageData.GetLength(0);
            int width = imageData.GetLength(1);
            int height = imageData.GetLength(2);

            if (labelData.Length != count)
            {
                throw new ArgumentException("Image/label count mismatch!");
            }

            string networkPath;
            if (miscArgs.Count > 0)
            {
                networkPath = miscArgs[0];
            }
            else
            {
                networkPath = "network.json";
            }

            Network network;
            if (File.Exists(networkPath))
            {
                using var stream = new FileStream(networkPath, FileMode.Open, FileAccess.Read);
                using var reader = new StreamReader(stream, Encoding.UTF8, leaveOpen: true);
                using var jsonReader = new JsonTextReader(reader)
                {
                    CloseInput = false
                };

                var data = serializer.Deserialize<NetworkData>(jsonReader);
                network = new Network(data);
            }
            else
            {
                network = new Network(new int[]
                {
                    width * height,
                    64,
                    16,
                    10
                });
            }

            var results = new ResultTable
            {
                ImageResults = new List<ImageResult>(),
                LayerSizes = network.LayerSizes
            };

            void serialize()
            {
                lock (network)
                {
                    Console.WriteLine("Serializing network...");

                    using var stream = new FileStream("network.json", FileMode.Create, FileAccess.Write);
                    using var writer = new StreamWriter(stream, Encoding.UTF8, leaveOpen: true);
                    using var jsonWriter = new JsonTextWriter(writer)
                    {
                        CloseOutput = false,
                    };

                    serializer!.Serialize(jsonWriter, network.Data);
                    Console.WriteLine("Finished serializing");
                }
            }

            Console.CancelKeyPress += (s, e) => serialize();
            if (train)
            {
                const int batchSize = 100;
                int batchCount = (int)Math.Floor((double)count / batchSize);

                Console.WriteLine($"Image count: {count}");
                Console.WriteLine($"Batch size: {batchSize}");
                Console.WriteLine($"Batch count: {batchCount} (rounded down)");

                while (train)
                {
                    Console.WriteLine("Creating index list...");
                    var imageIndices = new int[batchSize * batchCount];
                    for (int i = 0; i < imageIndices.Length; i++)
                    {
                        imageIndices[i] = i;
                    }

                    Console.WriteLine("Shuffling index list...");
                    imageIndices.Shuffle();

                    Console.WriteLine("Training on batches...");
                    for (int i = 0; i < batchCount; i++)
                    {
                        Console.WriteLine($"Creating batch {i + 1}...");

                        var batch = new LabeledData[batchSize];
                        for (int j = 0; j < batchSize; j++)
                        {
                            int index = imageIndices[i * batchSize + j];

                            var expectedOutput = new float[10];
                            for (int k = 0; k < expectedOutput.Length; k++)
                            {
                                expectedOutput[k] = labelData[index] == k ? 1f : 0f;
                            }

                            batch[j] = new LabeledData
                            {
                                Input = GetInputData(imageData, index),
                                ExpectedOutput = expectedOutput
                            };
                        }

                        lock (network)
                        {
                            Console.WriteLine($"Training network on batch {i + 1}...");
                            network.TrainOnBatch(batch, 0.1f, out float averageAbsoluteCost);

                            Console.WriteLine($"Average absolute cost: {averageAbsoluteCost}");
                            if (averageAbsoluteCost < breakThreshold)
                            {
                                Console.WriteLine($"Average is less than {breakThreshold} - breaking");

                                train = false;
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                double total = 0;
                for (int i = 0; i < count; i++)
                {
                    lock (network)
                    {
                        Console.WriteLine($"Image {i + 1}");
                        var inputData = GetInputData(imageData, i);

                        var begin = DateTime.Now;
                        var confidence = network.Evaluate(inputData);
                        var end = DateTime.Now;

                        var delta = end - begin;
                        var result = new ImageResult
                        {
                            ImageNumber = i + 1,
                            Value = labelData[i],
                            Confidence = new ValueConfidence[confidence.Length],
                            TimeElapsed = delta.TotalSeconds
                        };

                        var expectedValues = new float[confidence.Length];
                        for (int j = 0; j < confidence.Length; j++)
                        {
                            float expected = expectedValues[j] = labelData[i] == j ? 1f : 0f;
                            Console.WriteLine($"{j}: {confidence[j] * 100f}% (expected: {expected * 100f}%)");

                            result.Confidence[j] = new ValueConfidence
                            {
                                Value = j,
                                Confidence = confidence[j]
                            };
                        }

                        total += result.TimeElapsed;
                        Console.WriteLine($"Time elapsed: {result.TimeElapsed:0.#####} seconds ({total:0.#####} seconds total)");

                        results.ImageResults.Add(result);
                    }
                }

                using (var stream = new FileStream("results.json", FileMode.Create, FileAccess.Write))
                {
                    using var writer = new StreamWriter(stream, Encoding.UTF8, leaveOpen: true);
                    using var jsonWriter = new JsonTextWriter(writer)
                    {
                        CloseOutput = false,
                    };

                    serializer.Serialize(jsonWriter, results);
                }
            }

            serialize();
        }
    }
}