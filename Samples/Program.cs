using SlidingRank.FastOps;
using System;
using System.Threading.Tasks;
using VectorRAG.Net;

internal static class Program
{
    private static async Task Main()
    {
        // Deterministic local embedding model (no OpenAI key required)
        IEmbeddingModel model = new HashEmbeddingModel(dimension: 64);

        var lsh = new EmbeddingLshConfig(
        Bands: 12,
        BitsPerBand: 8,
        MaxCandidates: 1024,
        Seed: 1337);

        var dbOptions = new VectorRagDatabaseOptions
        {
            InitialCapacity = 2048,
            QueryCacheCapacity = 256,
            NormalizeVectorsOnAdd = true,
            NormalizeQueryOnSearch = true,
            DefaultChunking = new ChunkingOptions
            {
                Strategy = ChunkingStrategy.FixedChars,
                ChunkSize = 300,
                ChunkOverlap = 50
            }
        };

        var db = new VectorRAGDatabase(dimension: model.Dimension, lshConfig: lsh, options: dbOptions);

        await db.UpsertTextDocumentAsync(
        externalId: "doc:password_reset",
        text: "To reset your password,go to Settings -> Security -> Reset Password. " +
        "You will receive a confirmation code via email.",
        metadata: new DocumentMetadata { Department = "Support", IsActive = true, Source = "kb" },
        embeddingModel: model);

        await db.UpsertTextDocumentAsync(
        externalId: "doc:pricing_business",
        text: "Business pricing depends on monthly turnover. Contact Sales for a quote. " +
        "Discounts are available for enterprise customers.",
        metadata: new DocumentMetadata { Department = "Sales", IsActive = true, Source = "kb" },
        embeddingModel: model);

        var query = "How can I reset my password?";
        var qVec = await model.GenerateEmbeddingAsync(query);

        var results = db.Search(qVec, new SearchOptions
        {
            TopK = 5,
            UseHybrid = true,
            TextQuery = query,
            Alpha = 0.7f,
            Filter = md => md.IsActive && md.Department == "Support",
            GroupByParentDocument = true
        });

        Console.WriteLine($"Query:{query}");
        foreach (var r in results)
        {
            Console.WriteLine($"- {r.ExternalId} score={r.Score:0.000} evidenceChunk={r.EvidenceChunkIndex}");
            Console.WriteLine($" Evidence:{Trim(r.EvidenceText, 140)}");
        }

        // Persistence demo
        var snap = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "VectorRAG.Net.sample.snapshot");
        await db.SaveAsync(snap);
        await db.LoadAsync(snap);

        var m = db.GetMetrics();
        Console.WriteLine($"Metrics:active={m.RecordsActive}/{m.RecordsTotal},queries={m.QueriesTotal},avg={m.AvgQueryMs:0.00}ms");
    }

    private static string Trim(string? s, int max)
    {
        if (string.IsNullOrEmpty(s)) return "";
        if (s.Length <= max) return s;
        return s.Substring(0, max) + "...";
    }
}
