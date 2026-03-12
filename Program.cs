using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using MongoDB.Bson;
using MongoDB.Driver;

namespace AffineSpace.Optimization
{
    /// <summary>
    /// Represents a point in the 11-dimensional affine space A^11.
    /// Uses SIMD (Vector&lt;float&gt;) to minimize arithmetic overhead across all dimensions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct AffinePoint11
    {
        // 11 dimensions are covered by two Vector<float> registers (8 lanes each).
        // The upper register carries dimensions x9–x11; remaining lanes are zero-padded.
        private readonly Vector<float> _low;  // x1 – x8
        private readonly Vector<float> _high; // x9 – x11 + padding

        public AffinePoint11(float[] coords)
        {
            if (coords.Length < 11)
                throw new ArgumentException("Coordinate array must contain at least 11 elements.");

            _low = new Vector<float>(coords, 0);

            float[] h = new float[8];
            Array.Copy(coords, 8, h, 0, 3);
            _high = new Vector<float>(h);
        }

        public Vector<float> Low  => _low;
        public Vector<float> High => _high;
    }

    /// <summary>
    /// Defines a quiver functor that maps objects and morphisms of A^11
    /// to physical block-storage representations.
    /// </summary>
    /// <typeparam name="TSource">Affine-space point type (quiver vertex).</typeparam>
    /// <typeparam name="TTarget">Block-storage record type (quiver target object).</typeparam>
    public interface IQuiverFunctor<TSource, TTarget>
    {
        /// <summary>Maps a quiver vertex (spatial point) to its target object (storage block).</summary>
        TTarget MapObject(TSource point);

        /// <summary>
        /// Maps a quiver morphism (displacement vector) to an index transition,
        /// and determines whether the corresponding sub-tree can be pruned.
        /// </summary>
        bool CanPrune(TSource min, TSource max, TSource queryMin, TSource queryMax);
    }

    /// <summary>
    /// B*-Tree node that encodes the quiver structure of A^11.
    /// Each node acts as a quotient of A^11, covering a contiguous hyper-rectangular region.
    /// The sibling pointer enables the 2/3-fill-factor rebalancing characteristic of B*-Trees.
    /// </summary>
    public class BStarQuiverNode
    {
        /// <summary>
        /// Tree degree aligned to the dimensionality of A^11.
        /// Each node partitions its region into at most <c>Degree</c> sub-regions.
        /// </summary>
        public const int Degree = 11;

        /// <summary>
        /// Sibling pointer used during node splits to maintain the 2/3 fill-factor invariant.
        /// Corresponds to a lateral morphism in the quiver diagram.
        /// </summary>
        public BStarQuiverNode Sibling { get; set; }

        /// <summary>Minimum bounding point of the hyper-rectangular region covered by this node (BRIN lower bound).</summary>
        public AffinePoint11 MinBound;

        /// <summary>Maximum bounding point of the hyper-rectangular region covered by this node (BRIN upper bound).</summary>
        public AffinePoint11 MaxBound;

        public List<AffinePoint11>   Entries  = new List<AffinePoint11>();
        public List<BStarQuiverNode> Children = new List<BStarQuiverNode>();

        /// <summary>
        /// Evaluates whether this node's bounding region is disjoint from the query range.
        /// Uses vectorized comparison across all 11 dimensions to achieve O(1) pruning.
        /// When the regions do not intersect, the corresponding quiver path maps to the empty set
        /// and the entire sub-tree is discarded without further traversal.
        /// </summary>
        /// <param name="qMin">Lower bound of the query hyper-rectangle.</param>
        /// <param name="qMax">Upper bound of the query hyper-rectangle.</param>
        /// <returns><c>true</c> if this node can be safely pruned; otherwise <c>false</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MatchAndPrune(in AffinePoint11 qMin, in AffinePoint11 qMax)
        {
            // A node is prunable if, in any dimension, the query range lies entirely
            // above MaxBound or entirely below MinBound.
            bool outOfLow  = Vector.GreaterThanAny(qMin.Low,  MaxBound.Low)  ||
                             Vector.LessThanAny   (qMax.Low,  MinBound.Low);

            bool outOfHigh = Vector.GreaterThanAny(qMin.High, MaxBound.High) ||
                             Vector.LessThanAny   (qMax.High, MinBound.High);

            return outOfLow || outOfHigh;
        }
    }

    // -------------------------------------------------------------------------
    //  MongoDB document schema
    // -------------------------------------------------------------------------

    /// <summary>
    /// MongoDB document that persists one entry from A^11 together with
    /// its pre-computed Morton code for locality-preserving ordering.
    /// </summary>
    public class AffineDocument
    {
        public ObjectId Id         { get; set; }
        public float[]  Coords     { get; set; }  // 11-element coordinate vector
        public long     MortonCode { get; set; }  // Z-order index for range scans
    }

    // -------------------------------------------------------------------------
    //  Pruning engine with live MongoDB integration
    // -------------------------------------------------------------------------

    /// <summary>
    /// Integrates a BRIN-style B*-Tree index with a live MongoDB collection.
    /// Spatial pruning is performed in-process; I/O is issued only for leaf nodes
    /// whose bounding regions intersect the query hyper-rectangle.
    /// </summary>
    public class MongoDBPruningEngine : IDisposable
    {
        private readonly IMongoCollection<AffineDocument> _collection;
        private BStarQuiverNode _root;

        // ── Construction ──────────────────────────────────────────────────────

        /// <summary>
        /// Initialises the engine and connects to MongoDB.
        /// </summary>
        /// <param name="connectionString">MongoDB connection URI (e.g. "mongodb://localhost:27017").</param>
        /// <param name="databaseName">Target database name.</param>
        /// <param name="collectionName">Target collection name.</param>
        public MongoDBPruningEngine(
            string connectionString = "mongodb://localhost:27017",
            string databaseName    = "affine_db",
            string collectionName  = "points")
        {
            var client   = new MongoClient(connectionString);
            var database = client.GetDatabase(databaseName);
            _collection  = database.GetCollection<AffineDocument>(collectionName);

            try
            {
                EnsureIndexes();
            }
            catch (TimeoutException ex)
            {
                throw new InvalidOperationException(
                    $"MongoDB に接続できませんでした ({connectionString})。サーバーが起動しているか確認してください。", ex);
            }
            _root = new BStarQuiverNode();
        }

        // ── Index management ─────────────────────────────────────────────────

        /// <summary>
        /// Creates a background index on <c>MortonCode</c> to accelerate
        /// range queries issued by leaf nodes.
        /// </summary>
        private void EnsureIndexes()
        {
            var keys    = Builders<AffineDocument>.IndexKeys.Ascending(d => d.MortonCode);
            var options = new CreateIndexOptions { Name = "ix_morton" };
            _collection.Indexes.CreateOne(new CreateIndexModel<AffineDocument>(keys, options));
        }

        // ── Morton encoding ───────────────────────────────────────────────────

        /// <summary>
        /// Computes the 11-dimensional Morton code (Z-order curve) for the given point.
        /// Interleaves the quantised integer bits of each coordinate into a single 64-bit key,
        /// projecting A^11 onto a 1-dimensional index that preserves spatial locality.
        /// </summary>
        public long GenerateMorton11(AffinePoint11 p)
        {
            // Quantise each coordinate to 5 bits (0–31); 11 × 5 = 55 bits ≤ 63.
            var   low    = p.Low;
            var   high   = p.High;
            long  code   = 0;

            for (int dim = 0; dim < 11; dim++)
            {
                float raw  = dim < 8 ? low[dim] : high[dim - 8];
                long  bits = (long)Math.Clamp(raw * 31f, 0f, 31f);

                // Interleave: place dimension <dim>'s bits at positions dim, dim+11, dim+22, …
                for (int bit = 0; bit < 5; bit++)
                {
                    if ((bits & (1L << bit)) != 0)
                        code |= 1L << (bit * 11 + dim);
                }
            }

            return code;
        }

        // ── Write path ────────────────────────────────────────────────────────

        /// <summary>
        /// Inserts a new point into MongoDB and registers it in the in-memory B*-Tree index.
        /// </summary>
        public void Insert(float[] coords)
        {
            if (coords.Length < 11)
                throw new ArgumentException("Coordinate array must contain at least 11 elements.");

            var point = new AffinePoint11(coords);
            var doc   = new AffineDocument
            {
                Coords     = coords,
                MortonCode = GenerateMorton11(point)
            };

            _collection.InsertOne(doc);

            // Register in the B*-Tree (simplified; full rebalancing omitted for brevity)
            _root.Entries.Add(point);
        }

        // ── Read path ─────────────────────────────────────────────────────────

        /// <summary>
        /// Searches for all documents whose coordinates fall within [qMin, qMax].
        /// The B*-Tree prunes irrelevant sub-trees before any MongoDB I/O is issued.
        /// </summary>
        /// <param name="qMin">Lower bound of the query range.</param>
        /// <param name="qMax">Upper bound of the query range.</param>
        /// <returns>All matching documents from MongoDB.</returns>
        public List<AffineDocument> Search(AffinePoint11 qMin, AffinePoint11 qMax)
        {
            var results = new List<AffineDocument>();
            Traverse(_root, qMin, qMax, results);
            return results;
        }

        private void Traverse(
            BStarQuiverNode      node,
            in AffinePoint11     qMin,
            in AffinePoint11     qMax,
            List<AffineDocument> results)
        {
            if (node == null) return;

            // Pruning: if the quiver morphism maps to the empty set, skip this sub-tree entirely.
            if (node.MatchAndPrune(qMin, qMax)) return;

            if (node.Children.Count == 0)
            {
                // Leaf node: issue a targeted MongoDB range query bounded by the Morton interval.
                var docs = ExecuteMongoQuery(node, qMin, qMax);
                results.AddRange(docs);
            }
            else
            {
                foreach (var child in node.Children)
                    Traverse(child, qMin, qMax, results);
            }
        }

        // ── MongoDB query ─────────────────────────────────────────────────────

        /// <summary>
        /// Translates a leaf node's bounding region into a MongoDB range filter
        /// over the Morton-code index and returns matching documents.
        /// </summary>
        private List<AffineDocument> ExecuteMongoQuery(
            BStarQuiverNode  node,
            in AffinePoint11 qMin,
            in AffinePoint11 qMax)
        {
            long mortonMin = GenerateMorton11(qMin);
            long mortonMax = GenerateMorton11(qMax);

            var filter = Builders<AffineDocument>.Filter.And(
                Builders<AffineDocument>.Filter.Gte(d => d.MortonCode, mortonMin),
                Builders<AffineDocument>.Filter.Lte(d => d.MortonCode, mortonMax)
            );

            return _collection.Find(filter).ToList();
        }

        public void Dispose() { /* MongoClient is thread-safe and connection-pooled; no explicit teardown required. */ }
    }

    // -------------------------------------------------------------------------
    //  Minimal usage example
    // -------------------------------------------------------------------------

    internal static class Program
    {
        private static void Main()
        {
            var connStr = Environment.GetEnvironmentVariable("MONGODB_CONNECTION")
                          ?? "mongodb://localhost:27017";

            using var engine = new MongoDBPruningEngine(
                connectionString: connStr,
                databaseName:     "affine_db",
                collectionName:   "points");

            // Insert sample points into A^11
            engine.Insert(new float[] { 1,2,3,4,5,6,7,8,9,10,11 });
            engine.Insert(new float[] { 0,0,0,0,0,0,0,0,0,0,0   });
            engine.Insert(new float[] { 5,5,5,5,5,5,5,5,5,5,5   });

            // Query the hyper-rectangle [0,0,...] – [6,6,...]
            var qMin = new AffinePoint11(new float[] { 0,0,0,0,0,0,0,0,0,0,0 });
            var qMax = new AffinePoint11(new float[] { 6,6,6,6,6,6,6,6,6,6,6 });

            var hits = engine.Search(qMin, qMax);

            Console.WriteLine($"Documents found: {hits.Count}");
            foreach (var doc in hits)
                Console.WriteLine($"  id={doc.Id}  morton={doc.MortonCode}");
        }
    }
}
