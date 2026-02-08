using SlidingRank.FastOps;
using System;
using System.IO;
using System.Threading.Tasks;
using VectorRAG.Net;
using Xunit;

public sealed class VectorRAGDatabaseTests
{
    [Fact]
    public async Task Upsert_Search_Filter_Works()
    {
        IEmbeddingModel model = new HashEmbeddingModel(64);

        var db = CreateDb(model.Dimension);

        await db.UpsertTextDocumentAsync(
        "doc:a",
        "Reset password via Settings -> Security.",
        new DocumentMetadata { Department = "Support", IsActive = true },
        model);

        await db.UpsertTextDocumentAsync(
        "doc:b",
        "Pricing for business customers is handled by Sales.",
        new DocumentMetadata { Department = "Sales", IsActive = true },
        model);

        var q = "reset password";
        var qVec = await model.GenerateEmbeddingAsync(q);

        var results = db.Search(qVec, new SearchOptions
        {
            TopK = 10,
            UseHybrid = true,
            TextQuery = q,
            Alpha = 0.7f,
            Filter = md => md.Department == "Support"
        });

        Assert.NotEmpty(results);
        Assert.All(results, r => Assert.Equal("Support", r.Metadata?.Department));
    }

    [Fact]
    public async Task Save_Load_Roundtrip_Works()
    {
        IEmbeddingModel model = new HashEmbeddingModel(64);
        var db = CreateDb(model.Dimension);

        await db.UpsertTextDocumentAsync(
        "doc:1",
        "Hello world. This is a test document about password reset.",
        new DocumentMetadata { Department = "Support", IsActive = true },
        model);

        var dir = Path.Combine(Path.GetTempPath(), "VectorRAG.Net.tests." + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            await db.SaveAsync(dir);

            // load into same db instance (variant A config must match)
            await db.LoadAsync(dir);

            var qVec = await model.GenerateEmbeddingAsync("password reset");
            var results = db.Search(qVec, new SearchOptions
            {
                TopK = 5,
                UseHybrid = true,
                TextQuery = "password reset",
                Alpha = 0.7f
            });

            Assert.NotEmpty(results);
            Assert.Equal("doc:1", results[0].ExternalId);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch {  }
        }
    }

    private static VectorRAGDatabase CreateDb(int dim)
    {
        var lsh = new EmbeddingLshConfig(
        Bands: 12,
        BitsPerBand: 8,
        MaxCandidates: 1024,
        Seed: 1337);

        var opt = new VectorRagDatabaseOptions
        {
            InitialCapacity = 1024,
            QueryCacheCapacity = 0,
            NormalizeVectorsOnAdd = true,
            NormalizeQueryOnSearch = true,
            DefaultChunking = new ChunkingOptions
            {
                Strategy = ChunkingStrategy.FixedChars,
                ChunkSize = 300,
                ChunkOverlap = 50
            }
        };

        return new VectorRAGDatabase(dim, lsh, opt);
    }
}
