using BenchmarkDotNet.Attributes;
using SlidingRank.FastOps;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using VectorRAG.Net;

[MemoryDiagnoser]
public class VectorRagSearchBench
{
    private VectorRAGDatabase _db = default!;
    private IEmbeddingModel _model = default!;
    private float[] _query = default!;

    [Params(10_000)]
    public int Docs;

    [GlobalSetup]
    public async Task Setup()
    {
        _model = new HashEmbeddingModel(64);

        var lsh = new EmbeddingLshConfig(
        Bands: 12,
        BitsPerBand: 8,
        MaxCandidates: 2048,
        Seed: 1337);

        var opt = new VectorRagDatabaseOptions
        {
            InitialCapacity = Docs + 128,
            QueryCacheCapacity = 0,
            NormalizeVectorsOnAdd = true,
            NormalizeQueryOnSearch = true,
            DefaultChunking = new ChunkingOptions
            {
                Strategy = ChunkingStrategy.FixedChars,
                ChunkSize = 200,
                ChunkOverlap = 20
            }
        };

        _db = new VectorRAGDatabase(_model.Dimension, lsh, opt);

        // Add embeddings directly (fast setup; avoid calling OpenAI etc.)
        var batch = new List<DocumentEmbedding>(Docs);
        for (int i = 0; i < Docs; i++)
        {
            var text = (i % 5 == 0)
 ? $"Document {i} about password reset and security settings."
 : $"Document {i} about shipping returns and delivery policies.";
            var vec = await _model.GenerateEmbeddingAsync(text);

            batch.Add(new DocumentEmbedding
            {
                ExternalId = $"doc:{i}",
                ParentExternalId = $"doc:{i}",
                ChunkIndex = 0,
                Text = text,
                Vector = vec,
                Metadata = new DocumentMetadata { Department = (i % 2 == 0) ? "Support" : "Sales", IsActive = true }
            });
        }

        _db.AddBatch(batch);

        _query = await _model.GenerateEmbeddingAsync("reset password security");
    }

    [Benchmark]
    public int Search_VectorOnly_Top5()
    {
        var res = _db.Search(_query, new SearchOptions
        {
            TopK = 5,
            UseHybrid = false,
            GroupByParentDocument = true
        });
        return res.Count;
    }

    [Benchmark]
    public int Search_Hybrid_Top5()
    {
        var res = _db.Search(_query, new SearchOptions
        {
            TopK = 5,
            UseHybrid = true,
            TextQuery = "reset password security",
            Alpha = 0.7f,
            GroupByParentDocument = true
        });
        return res.Count;
    }
}
