using System;

namespace NeuralNetwork
{
    public static class MathUtilities
    {

        public static float Dot(float[] lhs, float[] rhs)
        {
            /*
            if (lhs.Length != rhs.Length)
            {
                throw new ArgumentException("Size mismatch!");
            }
            */

            float lhsLengthSquared = 0f;
            float rhsLengthSquared = 0f;
            float differenceLengthSquared = 0f;

            for (int i = 0; i < lhs.Length; i++)
            {
                lhsLengthSquared += MathF.Pow(lhs[i], 2f);
                rhsLengthSquared += MathF.Pow(rhs[i], 2f);
                differenceLengthSquared += MathF.Pow(lhs[i] - rhs[i], 2f);
            }

            return (lhsLengthSquared + rhsLengthSquared - differenceLengthSquared) / 2f;
        }

        public static float[] Composite(float[,] matrix, float[] vector)
        {
            /*
            if (matrix.GetLength(1) != vector.Length)
            {
                throw new ArgumentException("Size mismatch!");
            }
            */

            var result = new float[matrix.GetLength(0)];
            for (int i = 0; i < result.Length; i++)
            {
                var row = new float[vector.Length];
                for (int j = 0; j < row.Length; j++)
                {
                    row[j] = matrix[i, j];
                }

                result[i] = Dot(row, vector);
            }

            return result;
        }

        public static float[,] Composite(float[,] lhs, float[,] rhs)
        {
            int lhsColumns = lhs.GetLength(1);
            int rhsRows = rhs.GetLength(0);

            /*
            if (lhsColumns != rhsRows)
            {
                throw new ArgumentException("Size mismatch!");
            }
            */

            int rows = lhs.GetLength(0);
            int columns = rhs.GetLength(1);

            var result = new float[rows, columns];
            for (int y = 0; y < rows; y++)
            {
                var row = new float[lhsColumns];
                for (int x = 0; x < lhsColumns; x++)
                {
                    row[x] = lhs[y, x];
                }

                for (int x = 0; x < columns; x++)
                {
                    var column = new float[rhsRows];
                    for (int rhsY = 0; rhsY < rhsRows; rhsY++)
                    {
                        column[rhsY] = rhs[x, rhsY];
                    }

                    result[y, x] = Dot(row, column);
                }
            }

            return result;
        }

        public static float[,] Composite(float[] lhs, float[,] rhs)
        {
            /*
            if (rhs.GetLength(0) != 1)
            {
                throw new ArgumentException("Size mismatch!");
            }
            */

            int rows = lhs.Length;
            int columns = rhs.GetLength(1);

            var result = new float[rows, columns];
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    result[y, x] = lhs[y] * rhs[0, x];
                }
            }

            return result;
        }

        public static float[,] Transpose(float[,] matrix)
        {
            int rows = matrix.GetLength(1);
            int columns = matrix.GetLength(0);

            var result = new float[rows, columns];
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    result[y, x] = matrix[x, y];
                }
            }

            return result;
        }

        public static float[,] Matrix(float[] vector)
        {
            // vectors are 1 column wide matrices
            int rows = vector.Length;
            var matrix = new float[rows, 1];

            for (int i = 0; i < rows; i++)
            {
                matrix[i, 0] = vector[i];
            }

            return matrix;
        }

        public static float[,] Transpose(float[] vector)
        {
            int columns = vector.Length;
            var matrix = new float[1, columns];

            for (int i = 0; i < columns; i++)
            {
                matrix[0, i] = vector[i];
            }

            return matrix;
        }


        public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
        public static float SigmoidPrime(float x)
        {
            float sig = Sigmoid(x);
            return sig * (1f - sig);
        }

        public static float Cost(float x, float y) => MathF.Pow(x - y, 2f);

        // partial derivative
        // ???
        // https://github.com/mnielsen/neural-networks-and-deep-learning/blob/d15df08a69ed33ae16a2fff874f83b57a956172c/src/network.py#L129C14-L129C14
        public static float CostDerivative(float x, float y) => x - y;
    }
}