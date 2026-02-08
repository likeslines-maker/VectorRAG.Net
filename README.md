VectorRAG.Net

VectorRAG.Net is a .NET-native high-performance vector database library for semantic search and RAG (Retrieval-Augmented Generation).
Core search is based on Random Hyperplane LSH candidate generation with exact rerank by dot/cosine.

---

Benchmark results (local machine)

Environment:
OS:Windows 11 (10.0.26200.7705)
CPU:Intel Core i5-11400F (6C/12T)
.NET:8.0.23
BenchmarkDotNet:0.15.8

Workload:
Docs:10,000
Embedding dim:64 (synthetic deterministic embeddings for benchmarking)
TopK:5
Hybrid mode:Vector + BM25 (BM25 uses hashed term IDs; query-term DF cutoff enabled)
GroupByParentDocument:true

Results:

| Method | Docs | Mean | Allocated |
|---|---:|---:|---:|
| Search_VectorOnly_Top5 | 10000 | 15.15 μs | 5.69 KB |
| Search_Hybrid_Top5 | 10000 | 116.73 μs | 14.85 KB |

Notes:
Benchmarks were executed without an attached debugger (dotnet run -c Release).
Hybrid performance depends heavily on term distribution (worst-case queries with extremely common terms are intentionally mitigated).

---

Installation

Option  (NuGet)
```bash
dotnet add package VectorRAG.Net
```


---

Benchmark example

1) Install BenchmarkDotNet
```bash
dotnet add package BenchmarkDotNet
```

2) Benchmark program entry point (Program.cs)
```csharp
using BenchmarkDotNet.Running;

public static class Program
{
 public static void Main(string[] args)
 {
 BenchmarkRunner.Run<VectorRagSearchBench>();
 }
}
```

3) Benchmark class (VectorRagSearchBench.cs)
```csharp
using BenchmarkDotNet.Attributes;
using SlidingRank.FastOps;
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
 // Use a deterministic local embedding model for repeatable benchmarks
 _model = new HashEmbeddingModel(64);

 var lsh = new EmbeddingLshConfig(
 Bands:12,
 BitsPerBand:8,
 MaxCandidates:2048,
 Seed:1337);

 var opt = new VectorRagDatabaseOptions
 {
 InitialCapacity = Docs + 128,
 NormalizeVectorsOnAdd = true,
 NormalizeQueryOnSearch = true,
 DefaultChunking = new ChunkingOptions
 {
 Strategy = ChunkingStrategy.FixedChars,
 ChunkSize = 200,
 ChunkOverlap = 20
 }
 };

 _db = new VectorRAGDatabase(_model.Dimension,lsh,opt);

 // Add embeddings directly (fast setup)
 var batch = new List<DocumentEmbedding>(Docs);
 for (int i = 0; i < Docs; i++)
 {
 var text = (i % 5 == 0)
 ? $"Document {i} about password reset and security settings."
 :$"Document {i} about shipping returns and delivery policies.";

 var vec = await _model.GenerateEmbeddingAsync(text);

 batch.Add(new DocumentEmbedding
 {
 ExternalId = $"doc:{i}",
 ParentExternalId = $"doc:{i}",
 ChunkIndex = 0,
 Text = text,
 Vector = vec,
 Metadata = new DocumentMetadata { IsActive = true }
 });
 }

 _db.AddBatch(batch);
 _query = await _model.GenerateEmbeddingAsync("reset password security");
 }

 [Benchmark]
 public int Search_VectorOnly_Top5()
 {
 var res = _db.Search(_query,new SearchOptions
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
 var res = _db.Search(_query,new SearchOptions
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
```


Editions (commercial packaging)

| Edition | Price | For | Features |
|---|---:|---|---|
| Community | Free | Developers / Startups | Core engine + basic RAG |
| Professional | $499/mo | Companies up to 100 people | Persistence,metadata,convenience APIs |
| Enterprise | $1999/mo | Large orgs | Hybrid search,reranking (cross-encoder),SLA/support |

---

Commercial licensing & support
For commercial licensing,invoices,or support:
Contact:Arkhipov Vladimir
Email:vipvodu@yandex.ru
Telegram: @vipvodu
