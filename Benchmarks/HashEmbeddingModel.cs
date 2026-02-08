using System;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using VectorRAG.Net;

public sealed class HashEmbeddingModel : IEmbeddingModel
{
    public int Dimension { get; }

    public HashEmbeddingModel(int dimension)
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        Dimension = dimension;
    }

    public float[] GenerateEmbedding(string text)
    => GenerateEmbeddingAsync(text).GetAwaiter().GetResult();

    public Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        text ??= string.Empty;

        var vec = new float[Dimension];
        var tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i < tokens.Length; i++)
        {
            int h = StableHash32(tokens[i]);
            int idx = (h & 0x7fffffff) % Dimension;
            vec[idx] += ((h & 1) == 0) ? 1f : -1f;
        }

        Normalize(vec);
        return Task.FromResult(vec);
    }

    private static int StableHash32(string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        var hash = SHA256.HashData(bytes);
        return BitConverter.ToInt32(hash, 0);
    }

    private static void Normalize(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) sum += (double)v[i] * v[i];
        var norm = Math.Sqrt(sum);
        if (norm <= 0) return;
        float inv = (float)(1.0 / norm);
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }
}
