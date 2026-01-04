# Performance Hints

# 性能提示

Jeff Dean,
Sanjay Ghemawat

杰夫·迪恩，
桑杰·格玛瓦特

Original version: 2023/07/27, last updated: 2025/12/16

原始版本：2023/07/27，最后更新：2025/12/16

Expand all details
Collapse all details

展开所有细节
折叠所有细节

Over the years, we (Jeff & Sanjay) have done a fair bit of diving into
performance tuning of various pieces of code, and improving the
performance of our software  has been important from the very earliest days of Google, since it
lets us do more for more users. We wrote this document as a way of identifying
some general principles and specific techniques that we use when doing this sort
of work, and tried to pick illustrative source code changes (change lists, or
CLs) that provide examples of the various approaches and techniques. Most of the
concrete suggestions below reference C++ types and CLs, but the general
principles apply to other languages. The document focuses on general performance
tuning in the context of a single binary, and does not cover distributed systems
or machine learning (ML) hardware performance tuning (huge areas unto
themselves). We hope others will find this useful.

多年来，我们（Jeff & Sanjay）对各种代码进行了大量的
性能调优，并且从谷歌最早的日子开始，提高我们软件的
性能就一直很重要，因为它
让我们能为更多的用户做更多的事情。我们编写这份文档是为了确定
我们在做这类工作时使用的一些通用原则和具体技术，并试图挑选出一些说明性的源代码更改（变更列表，或
CLs），以提供各种方法和技术的示例。下面大部分的
具体建议都引用了C++类型和CLs，但通用
原则也适用于其他语言。本文档侧重于单个二进制文件上下文中的通用性能
调优，不涉及分布式系统
或机器学习（ML）硬件性能调优（这些本身就是巨大的领域）。我们希望其他人会觉得这很有用。

Many of the examples in the document have code fragments that demonstrate the
techniques (click the little triangles!). Note that some of these
code fragments mention various internal Google codebase abstractions. We have
included these anyway if we felt like the examples were self-contained enough to
be understandable to those unfamiliar with the details of those abstractions.

文档中的许多示例都有演示这些
技术的代码片段（点击小三角形！）。请注意，其中一些
代码片段提到了各种内部的谷歌代码库抽象。我们还
是包含了这些，因为我们觉得这些示例足够独立，对于
不熟悉这些抽象细节的人来说是可以理解的。

## The importance of thinking about performance

## 思考性能的重要性

Knuth is often quoted out of context as saying premature optimization is the
root of all evil. The
full quote reads: “We
should forget about small efficiencies, say about 97% of the time: premature
optimization is the root of all evil. Yet we should not pass up our
opportunities in that critical 3%.” This document is about that critical
3%, and a more compelling quote,
again from Knuth, reads:

高德纳（Knuth）经常被断章取义地引述说，过早的优化是
万恶之源。完整的引述是：“我们
应该忘记小的效率，比如说大约97%的时间：过早的
优化是万恶之源。然而，我们不应该放弃我们
在那关键的3%中的机会。”这份文件就是关于那关键的
3%，还有一个更有说服力的引述，
同样来自高德纳，内容如下：

Many people will say “let’s write down the code in as simple a way as possible
and deal with performance later when we can profile”. However, this approach is
often wrong:

许多人会说“让我们以最简单的方式编写代码
，稍后在我们可以进行性能分析时再处理性能”。然而，这种方法
通常是错误的：

<ol>
<li>If you disregard all performance concerns when developing a large system,
you will end up with a flat profile where there are no obvious hotspots
because performance is lost all over the place. It will be difficult to
figure out how to get started on performance improvements.</li>
<li>如果在开发大型系统时完全不考虑性能问题，
最终会得到一个平坦的性能剖析图，其中没有明显的热点，
因为性能损耗无处不在。这将很难
着手进行性能改进。</li>
<li>If you are developing a library that will be used by other people, the
people who will run into performance problems will be likely to be people
who cannot easily make performance improvements (they will have to
understand the details of code written by other people/teams, and have to
negotiate with them about the importance of performance optimizations).</li>
<li>It is harder to make significant changes to a system when it is in heavy
use.</li>
<li>It is also hard to tell if there are performance problems that can be solved
easily and so we end up with potentially expensive solutions like
over-replication or severe overprovisioning of a service to handle load
problems.</li>
</ol>

Instead, we suggest that when writing code, try to choose the faster alternative
if it does not impact readability/complexity of the code significantly.

相反，我们建议在编写代码时，如果不会显著影响代码的可读性/复杂性，就尽量选择更快的替代方案。

## Estimation

## 估算

If you can develop an intuition for how much performance might matter in the
code you are writing, you can make a more informed decision (e.g., how much
extra complexity is warranted in the name of performance). Some tips on
estimating performance while you are writing code:

如果你能对自己正在编写的代码的性能有多重要有一个直观的认识，你就能做出更明智的决定（例如，为了性能，多大的额外复杂性是合理的）。以下是一些在编写代码时估算性能的技巧：

<ul>
<li>Is it test code? If so, you need to worry mostly about the asymptotic
complexity of your algorithms and data structures. (Aside: development cycle
time matters, so avoid writing tests that take a long time to run.)</li>
<li>Is it code specific to an application? If so, try to figure out how much
performance matters for this piece of code. This is typically not very hard:
just figuring out whether code is initialization/setup code vs. code that
will end up on hot paths (e.g., processing every request in a service) is
often sufficient</li>
<li>Is it library code that will be used by many applications? In this case it
is hard to tell how sensitive it might become. This is where it becomes
especially important to follow some of the simple techniques described in
this document. For example, if you need to store a vector that usually has a
small number of elements, use an absl::InlinedVector instead of std::vector.
Such techniques are not very hard to follow and don’t add any non-local
complexity to the system. And if it turns out that the code you are writing
does end up using significant resources, it will be higher performance from
the start. And it will be easier to find the next thing to focus on when
looking at a profile.</li>
</ul>

You can do a slightly deeper analysis when picking between options with
potentially different performance characteristics by relying on
back of the envelope calculations.
Such calculations can quickly give a very rough estimate of the performance of
different alternatives, and the results can be used to discard some of the
alternatives without having to implement them.

Here is how such an estimation might work:

<ol>
<li>Estimate how many low-level operations of various kinds are required, e.g.,
number of disk seeks, number of network round-trips, bytes transmitted etc.</li>
<li>估算需要多少各种类型的低级操作，例如，
磁盘寻道次数、网络往返次数、传输的字节数等。</li>
<li>Multiply each kind of expensive operation with its rough cost, and add the
results together.</li>
<li>The preceding gives the cost of the system in terms of resource usage. If
you are interested in latency, and if the system has any concurrency, some
of the costs may overlap and you may have to do slightly more complicated
analysis to estimate the latency.</li>
</ol>

The following table, which is an updated version of a table from a
2007 talk at Stanford University
(video of the 2007 talk no longer exists, but there is a
video of a related 2011 Stanford talk that covers some of the same content)
may be useful since it lists the types of operations to consider, and their
rough cost:

The preceding table contains rough costs for some basic low-level operations.
You may find it useful to also track estimated costs for higher-level operations
relevant to your system. E.g., you might want to know the rough cost of a point
read from your SQL database, the latency of interacting with a Cloud service, or
the time to render a simple HTML page. If you don’t know the relevant cost of
different operations, you can’t do decent back-of-the-envelope calculations!

### Example: Time to quicksort a billion 4 byte numbers

### 示例：对十亿个4字节数字进行快速排序的时间

As a rough approximation, a good quicksort algorithm makes log(N) passes over an
array of size N. On each pass, the array contents will be streamed from memory
into the processor cache, and the partition code will compare each element once
to a pivot element. Let’s add up the dominant costs:

粗略地说，一个好的快速排序算法会对大小为N的数组进行log(N)次遍历。
在每次遍历中，数组内容将从内存流式传输到处理器缓存中，分区代码会将每个元素与一个枢轴元素进行一次比较。让我们加总一下主要成本：

<ol>
<li>Memory bandwidth: the array occupies 4 GB (4 bytes per number times a
billion numbers). Let’s assume ~16GB/s of memory bandwidth per core. That
means each pass will take ~0.25s. N is ~2^30, so we will make ~30 passes, so
the total cost of memory transfer will be ~7.5 seconds.</li>
<li>内存带宽：数组占用4 GB（每个数字4字节乘以
十亿个数字）。假设每个核心的内存带宽约为16GB/s。这
意味着每次遍历将花费约0.25秒。N约为2^30，因此我们将进行约30次遍历，所以
内存传输的总成本将约为7.5秒。</li>
<li>Branch mispredictions: we will do a total of N*log(N) comparisons, i.e., ~30
billion comparisons. Let’s assume that half of them (i.e., 15 billion) are
mispredicted. Multiplying by 5 ns per misprediction, we get a misprediction
cost of 75 seconds. We assume for this analysis that correctly predicted
branches are free.</li>
<li>分支预测错误：我们将进行总共N*log(N)次比较，即约300
亿次比较。假设其中一半（即150亿次）被错误预测。每次错误预测的成本为5纳秒，那么错误预测的
成本为75秒。在此分析中，我们假设正确预测的
分支是免费的。</li>
<li>Adding up the previous numbers, we get an estimate of ~82.5 seconds.</li>
</ol>

If necessary, we could refine our analysis to account for processor caches. This
refinement is probably not needed since branch mispredictions are the dominant
cost according to the analysis above, but we include it here anyway as another
example. Let’s assume we have a 32MB L3 cache, and that the cost of transferring
data from L3 cache to the processor is negligible. The L3 cache can hold 2^23
numbers, and therefore the last 22 passes can operate on the data resident in
the L3 cache (the 23rd last pass brings data into the L3 cache and the remaining
passes operate on that data.) That cuts down the memory transfer cost to 2.5
seconds (10 memory transfers of 4GB at 16GB/s) instead of 7.5 seconds (30 memory
transfers).

### Example: Time to generate a web page with 30 image thumbnails

Let’s compare two potential designs where the original images are stored on
disk, and each image is approximately 1MB in size.

<ol>
<li>Read the contents of the 30 images serially and generate a thumbnail for
each one. Each read takes one seek + one transfer, which adds up to 5ms for
the seek, and 10ms for the transfer, which adds up to 30 images times 15ms
per image, i.e., 450ms.</li>
<li>Read in parallel, assuming the images are spread evenly across K disks. The
previous resource usage estimate still holds, but latency will drop by
roughly a factor of K, ignoring variance (e.g, we will sometimes get unlucky
and one disk will have more than 1/Kth of the images we are reading).
Therefore if we are running on a distributed filesystem with hundreds of
disks, the expected latency will drop to ~15ms.</li>
<li>Let’s consider a variant where all images are on a single SSD. This changes
the sequential read performance to 20µs + 1ms per image, which adds up to
~30 ms overall.</li>
</ol>

## Measurement

## 测量

The preceding section gives some tips about how to think about performance when
writing code without worrying too much about how to measure the performance
impact of your choices. However, before you actually start making improvements,
or run into a tradeoff involving various things like performance, simplicity,
etc. you will want to measure or estimate potential performance benefits. Being
able to measure things effectively is the number one tool you’ll want to have in
your arsenal when doing performance-related work.

As an aside, it’s worth pointing out that profiling code that you’re unfamiliar
with can also be a good way of getting a general sense of the structure of the
codebase and how it operates. Examining the source code of heavily involved
routines in the dynamic call graph of a program can give you a high level sense
of “what happens” when running the code, which can then build your own
confidence in making performance-improving changes in slightly unfamiliar code.

### Profiling tools and tips

Many useful profiling tools are available. A useful tool to reach for first is
pprof since it gives
good high level performance information and is easy to use both locally and for
code running in production. Also try
perf if you want more
detailed insight into performance.

Some tips for profiling:

<ul>
<li>Build production binaries with appropriate debugging information and
optimization flags.</li>
<li>If you can, write a microbenchmark that covers the code you are
improving. Microbenchmarks improve turnaround time when making performance
improvements, help verify the impact of performance improvements, and can
help prevent future performance regressions. However microbenchmarks can
have pitfalls that make them non-representative of full system
performance. Useful libraries for writing microbenchmarks:
C++ Go Java.</li>
<li>Use a benchmark library to emit performance counter readings both
for better precision, and to get more insight into program behavior.</li>
<li>Lock contention can often artificially lower CPU usage. Some mutex
implementations provide support for profiling lock contention.</li>
<li>Use ML profilers for machine learning performance work .</li>
</ul>

### What to do when profiles are flat

You will often run into situations where your CPU profile is flat (there is no
obvious big contributor to slowness). This can often happen when all low-hanging
fruit has been picked. Here are some tips to consider if you find yourself in
this situation:

<ul>
<li>Don’t discount the value of many small optimizations! Making twenty separate
1% improvements in some subsystem is often eminently possible and
collectively mean a pretty sizable improvement (work of this flavor often
relies on having stable and high quality microbenchmarks). Some examples of
these sorts of changes are in the
changes that demonstrate multiple techniques
section.</li>
<li>Find loops closer to the top of call stacks (flame graph view of a CPU
profile can be helpful here). Potentially, the loop or the code it calls
could be restructured to be more efficient. Some code that initially built a
complicated graph structure incrementally by looping over nodes and edges of
the input was changed to build the graph structure in one shot by passing it
the entire input. This removed a bunch of internal checks that were
happening per edge in the initial code.</li>
<li>Take a step back and look for structural changes higher up in the call
stacks instead of concentrating on micro-optimizations. The techniques
listed under algorithmic improvements can be
useful when doing this.</li>
<li>Look for overly general code. Replace it with a customized or lower-level
implementation. E.g., if an application is repeatedly using a regular
expression match where a simple prefix match would suffice, consider
dropping the use of the regular expression.</li>
<li>Attempt to reduce the number of allocations:
get an allocation profile, and pick away at the highest
contributor to the number of allocations. This will have two effects: (1) It
will provide a direct reduction of the amount of time spent in the allocator
(and garbage collector for GC-ed languages) (2) There will often be a
reduction in cache misses since in a long running program using tcmalloc,
every allocation tends to go to a different cache line.</li>
<li>Gather other types of profiles, specially ones based on hardware performance
counters. Such profiles may point out functions that are encountering a high
cache miss rate. Techniques described in the
profiling tools and tips section can be
helpful.</li>
</ul>

## API considerations

## API 设计考量

Some of the techniques suggested below require changing data structures and
function signatures, which may be disruptive to callers. Try to organize code so
that the suggested performance improvements can be made inside an encapsulation
boundary without affecting public interfaces. This will be easier if your
modules are deep
(significant functionality accessed via a narrow interface).

Widely used APIs come under heavy pressure to add features. Be
careful when adding new features since these will constrain future
implementations and increase cost unnecessarily for users who don’t need the new
features. E.g., many C++ standard library containers promise iterator stability,
which in typical implementations increases the number of allocations
significantly, even though many users do not need pointer stability.

Some specific techniques are listed below. Consider carefully the performance
benefits vs. any API usability issues introduced by such changes.

下面列出了一些具体的技术。请仔细考虑这些
更改带来的性能优势与任何API可用性问题。

### Bulk APIs

### 批量API

Provide bulk ops to reduce expensive API boundary crossings or to take advantage
of algorithmic improvements.

提供批量操作以减少昂贵的API边界交叉或利用
算法改进。

### Add bulk MemoryManager::LookupMany interface.

### 添加批量的 MemoryManager::LookupMany 接口。

#### Problem Description
In addition to adding a bulk interface, this also simplified the signature for
the new bulk variant: it turns out clients only needed to know if all the keys
were found, so we can return a bool rather than a Status object.
memory_manager.h

#### 问题描述
除了添加批量接口外，这也简化了
新批量变体的签名：事实证明，客户端只需要知道是否找到了所有的键
，因此我们可以返回一个布尔值而不是一个Status对象。
memory_manager.h

#### Code Diff
```diff
- class MemoryManager {
-  public:
-   ...
-   util::StatusOr<LiveTensor> Lookup(const TensorIdProto& id);
+ class MemoryManager {
+  public:
+   ...
+   util::StatusOr<LiveTensor> Lookup(const TensorIdProto& id);
+ 
+   // Lookup the identified tensors
+   struct LookupKey {
+     ClientHandle client;
+     uint64 local_id;
+   };
+   bool LookupMany(absl::Span<const LookupKey> keys,
+                   absl::Span<tensorflow::Tensor> tensors);
```

### Add bulk ObjectStore::DeleteRefs API to amortize locking
overhead.

### 添加批量的 ObjectStore::DeleteRefs API 以摊销锁定
开销。

#### Problem Description
object_store.h
memory_tracking.cc

#### 问题描述
object_store.h
memory_tracking.cc

#### Code Diff
```diff
- template <typename T>
- class ObjectStore {
-  public:
-   ...
-   absl::Status DeleteRef(Ref);
+ template <typename T>
+ class ObjectStore {
+  public:
+   ...
+   absl::Status DeleteRef(Ref);
+ 
+   // Delete many references.  For each ref, if no other Refs point to the same
+   // object, the object will be deleted.  Returns non-OK on any error.
+   absl::Status DeleteRefs(absl::Span<const Ref> refs);
+   ...
+ template <typename T>
+ absl::Status ObjectStore<T>::DeleteRefs(absl::Span<const Ref> refs) {
+   util::Status result;
+   absl::MutexLock l(&mu_);
+   for (auto ref : refs) {
+     result.Update(DeleteRefLocked(ref));
+   }
+   return result;
+ }
```

### Use Floyd's
heap construction for efficient initialization.

### 使用弗洛伊德的
堆构造进行高效初始化。

#### Problem Description
Bulk initialization of a heap can be done in O(N) time, whereas adding one
element at a time and updating the heap property after each addition requires
O(N lg(N)) time.

Sometimes it is hard to change callers to use a new bulk API directly. In that
case it might be beneficial to use a bulk API internally and cache the results
for use in future non-bulk API calls:

### Cache block decode results for use in future calls.

### 缓存块解码结果以供将来调用。

#### Problem Description
Each lookup needs to decode a whole block of K entries. Store the decoded
entries in a cache and consult the cache on future lookups.
lexicon.cc

#### 问题描述
每次查找都需要解码一整个K个条目的块。将解码后的条目存储在缓存中，并在将来的查找中查询缓存。
lexicon.cc

#### Code Diff
```diff
- void GetTokenString(int pos, std::string* out) const {
-   ...
-   absl::FixedArray<LexiconEntry, 32> entries(pos + 1);
- 
-   // Decode all lexicon entries up to and including pos.
-   for (int i = 0; i <= pos; ++i) {
-     p = util::coding::TwoValuesVarint::Decode32(p, &entries[i].remaining,
-                                                 &entries[i].shared);
-     entries[i].remaining_str = p;
-     p += entries[i].remaining;  // remaining bytes trail each entry.
-   }
+ mutable std::vector<absl::InlinedVector<std::string, 16>> cache_;
+ ...
+ void GetTokenString(int pos, std::string* out) const {
+   ...
+   DCHECK_LT(skentry, cache_.size());
+   if (!cache_[skentry].empty()) {
+     *out = cache_[skentry][pos];
+     return;
+   }
+   ...
+   // Init cache.
+   ...
+   const char* prev = p;
+   for (int i = 0; i < block_sz; ++i) {
+     uint32 shared, remaining;
+     p = TwoValuesVarint::Decode32(p, &remaining, &shared);
+     auto& cur = cache_[skentry].emplace_back();
+     gtl::STLStringResizeUninitialized(&cur, remaining + shared);
+ 
+     std::memcpy(cur.data(), prev, shared);
+     std::memcpy(cur.data() + shared, p, remaining);
+     prev = cur.data();
+     p += remaining;
+   }
+   *out = cache_[skentry][pos];
```

### View types

### 视图类型

Prefer view types (e.g., std::string_view, std::Span<T>,
absl::FunctionRef<R(Args...)>) for function arguments (unless ownership of the
data is being transferred). These types reduce copying, and allow callers to
pick their own container types (e.g., one caller might use std::vector whereas
another one uses absl::InlinedVector).

### Pre-allocated/pre-computed arguments

### 预分配/预计算的参数

For frequently called routines, sometimes it is useful to allow higher-level
callers to pass in a data structure that they own or information that the called
routine needs that the client already has. This can avoid the low-level routine
being forced to allocate its own temporary data structure or recompute
already-available information.

### Add RPC_Stats::RecordRPC variant allowing client to pass in
already available WallTime value.

### 添加 RPC_Stats::RecordRPC 的变体，允许客户端传入
已经可用的 WallTime 值。

#### Problem Description
rpc-stats.h
clientchannel.cc

#### Code Diff
```diff
- static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m);
+ static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m,
+                       WallTime now);
```

### Thread-compatible vs. Thread-safe types

### 线程兼容与线程安全的类型

A type may be either thread-compatible (synchronized externally) or thread-safe
(synchronized internally). Most generally used types should be
thread-compatible. This way callers who do not need thread-safety don’t pay for
it.

### Make a class thread-compatible since callers are already
synchronized.

### 使一个类成为线程兼容的，因为调用者已经
同步了。

#### Problem Description
hitless-transfer-phase.cc
hitless-transfer-phase.cc

#### 问题描述
hitless-transfer-phase.cc
hitless-transfer-phase.cc

#### Code Diff
```diff
- TransferPhase HitlessTransferPhase::get() const {
-   static CallsiteMetrics cm("HitlessTransferPhase::get");
-   MonitoredMutexLock l(&cm, &mutex_);
-   return phase_;
- }
+ TransferPhase HitlessTransferPhase::get() const { return phase_; }
```

However if the typical use of a type needs synchronization, prefer to move the
synchronization inside the type. This allows the synchronization mechanism to be
tweaked as necessary to improve performance (e.g., sharding to reduce
contention) without affecting callers.

## Algorithmic improvements

## 算法改进

The most critical opportunities for performance improvements come from
algorithmic improvements, e.g., turning an O(N²) algorithm to O(N lg(N)) or
O(N), avoiding potentially exponential behavior, etc. These opportunities are
rare in stable code, but are worth paying attention to when writing new code. A
few examples that show such improvements to pre-existing code:

### Add nodes to cycle detection structure in reverse
post-order.

### 以后序遍历的逆序将节点添加到循环检测结构中。

#### Problem Description
We were previously adding graph nodes and edges one at a time to a
cycle-detection data structure, which required expensive work per edge. We now
add the entire graph in reverse post-order, which makes cycle-detection trivial.
graphcycles.h
graphcycles.cc
graph_partitioner.cc

#### Code Diff
```diff
- class GraphCycles : public util_graph::Graph {
-  public:
-   GraphCycles();
-   ~GraphCycles() override;
- 
-   using Node = util_graph::Node;
+ class GraphCycles : public util_graph::Graph {
+  public:
+   GraphCycles();
+   ~GraphCycles() override;
+ 
+   using Node = util_graph::Node;
+ 
+   // InitFrom adds all the nodes and edges from src, returning true if
+   // successful, false if a cycle is encountered.
+   // REQUIRES: no nodes and edges have been added to GraphCycles yet.
+   bool InitFrom(const util_graph::Graph& src);
```

### Replace the deadlock detection system built into a mutex
implementation with a better algorithm.

#### Problem Description
Replaced deadlock detection algorithm by one that is ~50x as fast and scales to
millions of mutexes without problem (the old algorithm relied on a 2K limit to
avoid a performance cliff). The new code is based on the following paper: A
dynamic topological sort algorithm for directed acyclic graphs David J. Pearce,
Paul H. J. Kelly Journal of Experimental Algorithmics (JEA) JEA Homepage archive
Volume 11, 2006, Article No. 1.7
The new algorithm takes O(|V|+|E|) space (instead of the O(|V|^2) bits needed by
the older algorithm). Lock-acquisition order graphs are very sparse, so this is
much less space. The algorithm is also quite simple: the core of it is ~100
lines of C++. Since the code now scales to much larger number of Mutexes, we
were able to relax an artificial 2K limit, which uncovered a number of latent
deadlocks in real programs.
Benchmark results: these were run in DEBUG mode since deadlock detection is
mainly enabled in debug mode. The benchmark argument (/2k etc.) is the number of
tracked nodes. At the default 2k limit of the old algorithm, the new algorithm
takes only 0.5 microseconds per InsertEdge compared to 22 microseconds for the
old algorithm. The new algorithm also easily scales to much larger graphs
without problems whereas the old algorithm keels over quickly.

### Replace an IntervalMap (with O(lg N) lookups) with a hash
table (O(1) lookups).

#### Problem Description
The initial code was using IntervalMap because it seemed like the right data
structure to support coalescing of adjacent blocks, but a hash table suffices
since the adjacent block can be found by a hash table lookup. This (plus other
changes in the CL) improve the performance of tpu::BestFitAllocator by ~4X.
best_fit_allocator.h

#### Code Diff
```diff
- using Block = gtl::IntervalMap<int64, BlockState>::Entry;
- ...
- // Map of pairs (address range, BlockState) with one entry for each allocation
- // covering the range [0, allocatable_range_end_).  Adjacent kFree and
- // kReserved blocks are coalesced. Adjacent kAllocated blocks are not
- // coalesced.
- gtl::IntervalMap<int64, BlockState> block_list_;
- 
- // Set of all free blocks sorted according to the allocation policy. Adjacent
- // free blocks are coalesced.
- std::set<Block, BlockSelector> free_list_;
+ // A faster hash function for offsets in the BlockTable
+ struct OffsetHash {
+   ABSL_ATTRIBUTE_ALWAYS_INLINE size_t operator()(int64 value) const {
+     uint64 m = value;
+     m *= uint64_t{0x9ddfea08eb382d69};
+     return static_cast<uint64_t>(m ^ (m >> 32));
+   }
+ };
+ 
+ // Hash table maps from block start address to block info.
+ // We include the length of the previous block in this info so we
+ // can find the preceding block to coalesce with.
+ struct HashTableEntry {
+   BlockState state;
+   int64 my_length;
+   int64 prev_length;  // Zero if there is no previous block.
+ };
+ using BlockTable = absl::flat_hash_map<int64, HashTableEntry, OffsetHash>;
```

### Replace sorted-list intersection (O(N log N)) with hash
table lookups (O(N)).

#### Problem Description
Old code to detect whether or not two nodes share a common source would get the
sources for each node in sorted order and then do a sorted intersection. The new
code places the sources for one node in a hash-table and then iterates over the
other node's sources checking the hash-table.

### Implement good hash function so that things are O(1)
instead of O(N).

#### Problem Description
location.h
location.cc

#### Code Diff
```diff
- // Hasher for Location objects.
- struct LocationHash {
-   size_t operator()(const Location* key) const {
-     return key != nullptr ? util_hash::Hash(key->address()) : 0;
-   }
- };
+ size_t HashLocation(const Location& loc);
+ ...
+ struct LocationHash {
+   size_t operator()(const Location* key) const {
+     return key != nullptr ? HashLocation(*key) : 0;
+   }
+ };
```

## Better memory representation

## 更好的内存表示

Careful consideration of memory footprint and cache footprint of important data
structures can often yield big savings. The data structures below focus on
supporting common operations by touching fewer cache lines. Care taken here can
(a) avoid expensive cache misses (b) reduce memory bus traffic, which speeds up
both the program in question and anything else running on the same machine. They
rely on some common techniques you may find useful when designing your own data
structures.

仔细考虑重要数据结构的内存占用和缓存占用通常可以节省大量开销。下面的数据结构侧重于通过接触更少的缓存行来支持常见操作。这里的谨慎可以(a)避免昂贵的缓存未命中(b)减少内存总线流量，从而加快相关程序和在同一台机器上运行的任何其他程序的速度。它们依赖于一些常见的技术，在设计自己的数据结构时可能会发现这些技术很有用。

### Compact data structures

Use compact representations for data that will be accessed often or that
comprises a large portion of the application’s memory usage. A compact
representation can significantly reduce memory usage and improve performance by
touching fewer cache lines and reducing memory bus bandwidth usage. However,
watch out for cache-line contention.

### Memory layout

Carefully consider the memory layout of types that have a large memory or cache
footprint.

<ul>
<li>Reorder fields to reduce padding between fields with different alignment
requirements (see
class layout discussion).</li>
<li>Use smaller numeric types where the stored data will fit in the smaller
type.</li>
<li>Enum values sometimes take up a whole word unless you’re careful. Consider
using a smaller representation (e.g., use enum class OpType : uint8_t { ...
} instead of enum class OpType { ... }).</li>
<li>Order fields so that fields that are frequently accessed together are closer
to each other – this will reduce the number of cache lines touched on common
operations.</li>
<li>Place hot read-only fields away from hot mutable fields so that writes to
the mutable fields do not cause the read-only fields to be evicted from
nearby caches.</li>
<li>Move cold data so it does not live next to hot data, either by placing the
cold data at the end of the struct, or behind a level of indirection, or in
a separate array.</li>
<li>Consider packing things into fewer bytes by using bit and byte-level
encoding. This can be complicated, so only do this when the data under
question is encapsulated inside a well-tested module, and the overall
reduction of memory usage is significant. Furthermore, watch out for side
effects like under-alignment of frequently used data, or more expensive code
for accessing packed representations. Validate such changes using
benchmarks.</li>
</ul>

### Indices instead of pointers

On modern 64-bit machines, pointers take up 64 bits. If you have a pointer-rich
data structure, you can easily chew up lots of memory with indirections of T*.
Instead, consider using integer indices into an array T[] or other data
structure. Not only will the references be smaller (if the number of indices is
small enough to fit in 32 or fewer bits), but the storage for all the T[]
elements will be contiguous, often leading to better cache locality.

### Batched storage

Avoid data structures that allocate a separate object per stored element (e.g.,
std::map, std::unordered_map in C++). Instead, consider types that use
chunked or flat representations to store multiple elements in close proximity in
memory (e.g., std::vector, absl::flat_hash_{map,set} in C++). Such types
tend to have much better cache behavior. Furthermore, they encounter less
allocator overhead.

One useful technique is to partition elements into chunks where each chunk can
hold a fixed number of elements. This technique can reduce the cache footprint
of a data structure significantly while preserving good asymptotic behavior.

For some data structures, a single chunk suffices to hold all elements (e.g.,
strings and vectors). Other types (e.g., absl::flat_hash_map) also use this
technique.

### Inlined storage

Some container types are optimized for storing a small number of elements. These
types provide space for a small number of elements at the top level and
completely avoid allocations when the number of elements is small. This can be
very helpful when instances of such types are constructed often (e.g., as stack
variables in frequently executed code), or if many instances are live at the
same time. If a container will typically contain a small number of elements
consider using one of the inlined storage types, e.g., InlinedVector.

Caveat: if sizeof(T) is large, inlined storage containers may not be the best
choice since the inlined backing store will be large.

### Unnecessarily nested maps

Sometimes a nested map data structure can be replaced with a single-level map
with a compound key. This can reduce the cost of lookups and insertions
significantly.

### Reduce allocations and improve cache footprint by
converting btree<a,btree<b,c>> to btree<pair<a,b>,c>.

#### Problem Description
graph_splitter.cc

#### Code Diff
```diff
- absl::btree_map<std::string, absl::btree_map<std::string, OpDef>> ops;
+ // The btree maps from {package_name, op_name} to its const Opdef*.
+ absl::btree_map<std::pair<absl::string_view, absl::string_view>,
+                 const OpDef*>
+     ops;
```

Caveat: if the first map key is big, it might be better to stick with nested
maps:

### Switch to a nested map leads to 76% performance
improvement in microbenchmark.

#### Problem Description
We previously had a single-level hash table where the key consisted of a
(string) path and some other numeric sub-keys. Each path occurred in
approximately 1000 keys on average. We split the hash table into two levels
where the first level was keyed by the path and each second level hash table
kept just the sub-key to data mapping for a particular path. This reduced the
memory usage for storing paths by a factor of 1000, and also sped up accesses
where many sub-keys for the same path were accessed together.

### Arenas

Arenas can help reduce memory allocation cost, but they also have the benefit of
packing together independently allocated items next to each other, typically in
fewer cache lines, and eliminating most destruction costs. They are likely most
effective for complex data structures with many sub-objects. Consider providing
an appropriate initial size for the arena since that can help reduce
allocations.

Caveat: it is easy to misuse arenas by putting too many short-lived objects in a
long-lived arena, which can unnecessarily bloat memory footprint.

### Arrays instead of maps

If the domain of a map can be represented by a small integer or is an enum, or
if the map will have very few elements, the map can sometimes be replaced by an
array or a vector of some form.

### Use an array instead of flat_map.

#### Problem Description
rtp_controller.h

#### Code Diff
```diff
- const gtl::flat_map<int, int> payload_type_to_clock_frequency_;
+ // A map (implemented as a simple array) indexed by payload_type to clock freq
+ // for that paylaod type (or 0)
+ struct PayloadTypeToClockRateMap {
+   int map[128];
+ };
+ ...
+ const PayloadTypeToClockRateMap payload_type_to_clock_frequency_;
```

### Bit vectors instead of sets

If the domain of a set can be represented by a small integer, the set can be
replaced with a bit vector (InlinedBitVector is often a good choice). Set
operations can also be nicely efficient on these representations using bitwise
boolean operations (OR for union, AND for intersection, etc.).

### Spanner placement system. Replace
dense_hash_set<ZoneId> with a bit-vector with one bit per zone.

#### Problem Description
zone_set.h
Benchmark results:

#### Code Diff
```diff
- class ZoneSet: public dense_hash_set<ZoneId> {
-  public:
-   ...
-   bool Contains(ZoneId zone) const {
-     return count(zone) > 0;
-   }
+ class ZoneSet {
+   ...
+   // Returns true iff "zone" is contained in the set
+   bool ContainsZone(ZoneId zone) const {
+     return zone < b_.size() && b_.get_bit(zone);
+   }
+   ...
+  private:
+   int size_;          // Number of zones inserted
+   util::bitmap::InlinedBitVector<256> b_;
```

### Use bit matrix to keep track of reachability properties
between operands instead of hash table.

#### Problem Description
hlo_computation.h

#### Code Diff
```diff
- using TransitiveOperandMap =
-     std::unordered_map<const HloInstruction*,
-                        std::unordered_set<const HloInstruction*>>;
+ class HloComputation::ReachabilityMap {
+   ...
+   // dense id assignment from HloInstruction* to number
+   tensorflow::gtl::FlatMap<const HloInstruction*, int> ids_;
+   // matrix_(a,b) is true iff b is reachable from a
+   tensorflow::core::Bitmap matrix_;
+ };
```

## Reduce allocations

## 减少分配

Memory allocation adds costs:

<ol>
<li>It increases the time spent in the allocator.</li>
<li>Newly-allocated objects may require expensive initialization and sometimes
corresponding expensive destruction when no longer needed.</li>
<li>Every allocation tends to be on a new cache line and therefore data spread
across many independent allocations will have a larger cache footprint than
data spread across fewer allocations.</li>
</ol>

Garbage-collection runtimes sometimes obviate issue #3 by placing consecutive
allocations sequentially in memory.

### Avoid unnecessary allocations

### 避免不必要的分配

### Reducing allocations increases benchmark throughput by
21%.

#### Problem Description
memory_manager.cc

#### Code Diff
```diff
- LiveTensor::LiveTensor(tf::Tensor t, std::shared_ptr<const DeviceInfo> dinfo,
-                        bool is_batched)
-     : tensor(std::move(t)),
-       device_info(dinfo ? std::move(dinfo) : std::make_shared<DeviceInfo>()),
-       is_batched(is_batched) {
+ static const std::shared_ptr<DeviceInfo>& empty_device_info() {
+   static std::shared_ptr<DeviceInfo>* result =
+       new std::shared_ptr<DeviceInfo>(new DeviceInfo);
+   return *result;
+ }
+ 
+ LiveTensor::LiveTensor(tf::Tensor t, std::shared_ptr<const DeviceInfo> dinfo,
+                        bool is_batched)
+     : tensor(std::move(t)), is_batched(is_batched) {
+   if (dinfo) {
+     device_info = std::move(dinfo);
+   } else {
+     device_info = empty_device_info();
+   }
```

### Use statically-allocated zero vector when possible rather
than allocating a vector and filling it with zeroes.

#### Problem Description
embedding_executor_8bit.cc

Also, prefer stack allocation over heap allocation when object lifetime is
bounded by the scope (although be careful with stack frame sizes for large
objects).

### Resize or reserve containers

### 调整容器大小或保留空间

When the maximum or expected maximum size of a vector (or some other container
types) is known in advance, pre-size the container’s backing store (e.g., using
resize or reserve in C++).

### Pre-size a vector and fill it in, rather than N push_back
operations.

### 预先调整向量大小并填充它，而不是进行N次push_back
操作。

#### Problem Description
indexblockdecoder.cc

#### Code Diff
```diff
- for (int i = 0; i < ndocs-1; i++) {
-   uint32 delta;
-   ERRORCHECK(b->GetRice(rice_base, &delta));
-   docs_.push_back(DocId(my_shard_ + (base + delta) * num_shards_));
-   base = base + delta + 1;
- }
- docs_.push_back(last_docid_);
+ docs_.resize(ndocs);
+ DocId* docptr = &docs_[0];
+ for (int i = 0; i < ndocs-1; i++) {
+   uint32 delta;
+   ERRORCHECK(b.GetRice(rice_base, &delta));
+   *docptr = DocId(my_shard_ + (base + delta) * num_shards_);
+   docptr++;
+   base = base + delta + 1;
+ }
+ *docptr = last_docid_;
```

Caveat: Do not use resize or reserve to grow one element at a time since
that may lead to quadratic behavior. Also, if element construction is expensive,
prefer an initial reserve call followed by several push_back or
emplace_back calls instead of an initial resize since that will double the
number of constructor calls.

### Avoid copying when possible

### 尽可能避免复制

<ul>
<li>Prefer moving to copying data structures when possible.</li>
<li>If lifetime is not an issue, store pointers or indices instead of copies of
objects in transient data structures. E.g., if a local map is used to select
a set of protos from an incoming list of protos, we can make the map store
just pointers to the incoming protos instead of copying potentially deeply
nested data. Another common example is sorting a vector of indices rather
than sorting a vector of large objects directly since the latter would incur
significant copying/moving costs.</li>
</ul>

### Avoid an extra copy when receiving a tensor via gRPC.

### 通过gRPC接收张量时避免额外的复制。

#### Problem Description
A benchmark that sends around 400KB tensors speeds up by ~10-15%:

#### 问题描述
发送约400KB张量的基准测试速度提高了约10-15%：

### Move large options structure rather than copying it.

### 移动大的选项结构而不是复制它。

#### Problem Description
index.cc

#### 问题描述
index.cc

#### Code Diff
```diff
- return search_iterators::DocPLIteratorFactory::Create(opts);
+ return search_iterators::DocPLIteratorFactory::Create(std::move(opts));
```

### Use std::sort instead of std::stable_sort, which avoids
an internal copy inside the stable sort implementation.

### 使用 std::sort 而不是 std::stable_sort，以避免
稳定排序实现中的内部复制。

#### Problem Description
encoded-vector-hits.h

#### Code Diff
```diff
- std::stable_sort(hits_.begin(), hits_.end(),
-                  gtl::OrderByField(&HitWithPayloadOffset::docid));
+ struct HitWithPayloadOffset {
+   search_iterators::LocalDocId64 docid;
+   int first_payload_offset;  // offset into the payload vector.
+   int num_payloads;
+ 
+   bool operator<(const HitWithPayloadOffset& other) const {
+     return (docid < other.docid) ||
+            (docid == other.docid &&
+             first_payload_offset < other.first_payload_offset);
+   }
+ };
+     ...
+     std::sort(hits_.begin(), hits_.end());
```

### Reuse temporary objects

### 重用临时对象

A container or an object declared inside a loop will be recreated on every loop
iteration. This can lead to expensive construction, destruction, and resizing.
Hoisting the declaration outside the loop enables reuse and can provide a
significant performance boost. (Compilers are often unable to do such hoisting
on their own due to language semantics or their inability to ensure program
equivalence.)

### Hoist variable definition outside of loop iteration.

### 将变量定义提升到循环迭代之外。

#### Problem Description
autofdo_profile_utils.h

#### Code Diff
```diff
- auto iterator = absl::WrapUnique(sstable->GetIterator());
- while (!iterator->done()) {
-   T profile;
-   if (!profile.ParseFromString(iterator->value_view())) {
-     return absl::InternalError(
-         "Failed to parse mem_block to specified profile type.");
-   }
-   ...
-   iterator->Next();
- }
+ auto iterator = absl::WrapUnique(sstable->GetIterator());
+ T profile;
+ while (!iterator->done()) {
+   if (!profile.ParseFromString(iterator->value_view())) {
+     return absl::InternalError(
+         "Failed to parse mem_block to specified profile type.");
+   }
+   ...
+   iterator->Next();
+ }
```

### Define a protobuf variable outside a loop so that its
allocated storage can be reused across loop iterations.

#### Problem Description
stats-router.cc

#### Code Diff
```diff
- for (auto& r : routers_to_update) {
-   ...
-   ResourceRecord record;
-   {
-     MutexLock agg_lock(r.agg->mutex());
-     r.agg->AddResourceRecordUsages(measure_indices, &record);
-   }
-   ...
- }
+ ResourceRecord record;
+ for (auto& r : routers_to_update) {
+   ...
+   record.Clear();
+   {
+     MutexLock agg_lock(r.agg->mutex());
+     r.agg->AddResourceRecordUsages(measure_indices, &record);
+   }
+   ...
+ }
```

### Serialize to same std::string repeatedly.

#### Problem Description
program_rep.cc

#### Code Diff
```diff
- std::string DeterministicSerialization(const proto2::Message& m) {
-   std::string result;
-   proto2::io::StringOutputStream sink(&result);
-   proto2::io::CodedOutputStream out(&sink);
-   out.SetSerializationDeterministic(true);
-   m.SerializePartialToCodedStream(&out);
-   return result;
- }
+ absl::string_view DeterministicSerializationTo(const proto2::Message& m,
+                                                std::string* scratch) {
+   scratch->clear();
+   proto2::io::StringOutputStream sink(scratch);
+   proto2::io::CodedOutputStream out(&sink);
+   out.SetSerializationDeterministic(true);
+   m.SerializePartialToCodedStream(&out);
+   return absl::string_view(*scratch);
+ }
```

Caveat: protobuf, string, vector, containers etc. tend to grow to the size of
the largest value ever stored in them. Therefore reconstructing them
periodically (e.g., after every N uses) can help reduce memory requirements and
reinitialization costs.

## Avoid unnecessary work

## 避免不必要的工作

Perhaps one of the most effective categories of improving performance is
avoiding work you don’t have to do. This can take many forms, including creating
specialized paths through code for common cases that avoid more general
expensive computation, precomputation, deferring work until it is really needed,
hoisting work into less-frequently executed pieces of code, and other similar
approaches. Below are many examples of this general approach, categorized into a
few representative categories.

### Fast paths for common cases

Often, code is written to cover all cases, but some subset of the cases are much
simpler and more common than others. E.g., vector::push_back usually has
enough space for the new element, but contains code to resize the underlying
storage when it does not. Some attention paid to the structure of code can help
make the common simple case faster without hurting uncommon case performance
significantly.

### Make fast path cover more common cases.

#### Problem Description
Add handling of trailing single ASCII bytes, rather than only handling multiples
of four bytes with this routine. This avoids calling the slower generic routine
for all-ASCII strings that are, for example, 5 bytes.
utf8statetable.cc

#### Code Diff
```diff
- // Scan a UTF-8 stringpiece based on state table.
- // Always scan complete UTF-8 characters
- // Set number of bytes scanned. Return reason for exiting
- // OPTIMIZED for case of 7-bit ASCII 0000..007f all valid
- int UTF8GenericScanFastAscii(const UTF8ScanObj* st, absl::string_view str,
-                              int* bytes_consumed) {
-                              ...
-   int exit_reason;
-   do {
-     //  Skip 8 bytes of ASCII at a whack; no endianness issue
-     while ((src_limit - src >= 8) &&
-            (((UNALIGNED_LOAD32(src + 0) | UNALIGNED_LOAD32(src + 4)) &
-              0x80808080) == 0)) {
-       src += 8;
-     }
-     //  Run state table on the rest
-     int rest_consumed;
-     exit_reason = UTF8GenericScan(
-         st, absl::ClippedSubstr(str, src - initial_src), &rest_consumed);
-     src += rest_consumed;
-   } while (exit_reason == kExitDoAgain);
- 
-   *bytes_consumed = src - initial_src;
-   return exit_reason;
- }
+ // Scan a UTF-8 stringpiece based on state table.
+ // Always scan complete UTF-8 characters
+ // Set number of bytes scanned. Return reason for exiting
+ // OPTIMIZED for case of 7-bit ASCII 0000..007f all valid
+ int UTF8GenericScanFastAscii(const UTF8ScanObj* st, absl::string_view str,
+                              int* bytes_consumed) {
+                              ...
+   int exit_reason = kExitOK;
+   do {
+     //  Skip 8 bytes of ASCII at a whack; no endianness issue
+     while ((src_limit - src >= 8) &&
+            (((UNALIGNED_LOAD32(src + 0) | UNALIGNED_LOAD32(src + 4)) &
+              0x80808080) == 0)) {
+       src += 8;
+     }
+     while (src < src_limit && Is7BitAscii(*src)) { // Skip ASCII bytes
+       src++;
+     }
+     if (src < src_limit) {
+       //  Run state table on the rest
+       int rest_consumed;
+       exit_reason = UTF8GenericScan(
+           st, absl::ClippedSubstr(str, src - initial_src), &rest_consumed);
+       src += rest_consumed;
+     }
+   } while (exit_reason == kExitDoAgain);
+ 
+   *bytes_consumed = src - initial_src;
+   return exit_reason;
+ }
```

### Simpler fast paths for InlinedVector.

#### Problem Description
inlined_vector.h

#### Code Diff
```diff
- auto Storage<T, N, A>::Resize(ValueAdapter values, size_type new_size) -> void {
-   StorageView storage_view = MakeStorageView();
- 
-   IteratorValueAdapter<MoveIterator> move_values(
-       MoveIterator(storage_view.data));
- 
-   AllocationTransaction allocation_tx(GetAllocPtr());
-   ConstructionTransaction construction_tx(GetAllocPtr());
- 
-   absl::Span<value_type> construct_loop;
-   absl::Span<value_type> move_construct_loop;
-   absl::Span<value_type> destroy_loop;
- 
-   if (new_size > storage_view.capacity) {
-   ...
-   } else if (new_size > storage_view.size) {
-     construct_loop = {storage_view.data + storage_view.size,
-                       new_size - storage_view.size};
-   } else {
-     destroy_loop = {storage_view.data + new_size, storage_view.size - new_size};
-   }
+ auto Storage<T, N, A>::Resize(ValueAdapter values, size_type new_size) -> void {
+   StorageView storage_view = MakeStorageView();
+   auto* const base = storage_view.data;
+   const size_type size = storage_view.size;
+   auto* alloc = GetAllocPtr();
+   if (new_size <= size) {
+     // Destroy extra old elements.
+     inlined_vector_internal::DestroyElements(alloc, base + new_size,
+                                              size - new_size);
+   } else if (new_size <= storage_view.capacity) {
+     // Construct new elements in place.
+     inlined_vector_internal::ConstructElements(alloc, base + size, &values,
+                                                new_size - size);
+   } else {
+   ...
+   }
```

### Fast path for common cases of initializing 1-D to 4-D
tensors.

#### Problem Description
tensor_shape.cc

#### Code Diff
```diff
- template <class Shape>
- TensorShapeBase<Shape>::TensorShapeBase(gtl::ArraySlice<int64> dim_sizes) {
-   set_tag(REP16);
-   set_data_type(DT_INVALID);
-   set_ndims_byte(0);
-   set_num_elements(1);
-   for (int64 s : dim_sizes) {
-     AddDim(internal::SubtleMustCopy(s));
-   }
- }
+ template <class Shape>
+ void TensorShapeBase<Shape>::InitDims(gtl::ArraySlice<int64> dim_sizes) {
+   DCHECK_EQ(tag(), REP16);
+ 
+   // Allow sizes that are under kint64max^0.25 so that 4-way multiplication
+   // below cannot overflow.
+   static const uint64 kMaxSmall = 0xd744;
+   static_assert(kMaxSmall * kMaxSmall * kMaxSmall * kMaxSmall <= kint64max,
+                 "bad overflow check");
+   bool large_size = false;
+   for (auto s : dim_sizes) {
+     if (s > kMaxSmall) {
+       large_size = true;
+       break;
+     }
+   }
+ 
+   if (!large_size) {
+     // Every size fits in 16 bits; use fast-paths for dims in {1,2,3,4}.
+     uint16* dst = as16()->dims_;
+     switch (dim_sizes.size()) {
+       case 1: {
+         set_ndims_byte(1);
+         const int64 size = dim_sizes[0];
+         const bool neg = Set16(kIsPartial, dst, 0, size);
+         set_num_elements(neg ? -1 : size);
+         return;
+       }
+       case 2: {
+         set_ndims_byte(2);
+         const int64 size0 = dim_sizes[0];
+         const int64 size1 = dim_sizes[1];
+         bool neg = Set16(kIsPartial, dst, 0, size0);
+         neg |= Set16(kIsPartial, dst, 1, size1);
+         set_num_elements(neg ? -1 : (size0 * size1));
+         return;
+       }
+       case 3: {
+       ...
+       }
+       case 4: {
+       ...
+       }
+     }
+   }
+ 
+   set_ndims_byte(0);
+   set_num_elements(1);
+   for (int64 s : dim_sizes) {
+     AddDim(internal::SubtleMustCopy(s));
+   }
+ }
```

### Make varint parser fast path cover just the 1-byte case,
instead of covering 1-byte and 2-byte cases.

#### Problem Description
Reducing the size of the (inlined) fast path reduces code size and icache
pressure, which leads to improved performance.
parse_context.h
parse_context.cc

#### Code Diff
```diff
- template <typename T>
- PROTOBUF_NODISCARD const char* VarintParse(const char* p, T* out) {
-   auto ptr = reinterpret_cast<const uint8_t*>(p);
-   uint32_t res = ptr[0];
-   if (!(res & 0x80)) {
-     *out = res;
-     return p + 1;
-   }
-   uint32_t byte = ptr[1];
-   res += (byte - 1) << 7;
-   if (!(byte & 0x80)) {
-     *out = res;
-     return p + 2;
-   }
-   return VarintParseSlow(p, res, out);
- }
+ template <typename T>
+ PROTOBUF_NODISCARD const char* VarintParse(const char* p, T* out) {
+   auto ptr = reinterpret_cast<const uint8_t*>(p);
+   uint32_t res = ptr[0];
+   if (!(res & 0x80)) {
+     *out = res;
+     return p + 1;
+   }
+   return VarintParseSlow(p, res, out);
+ }
```

### Skip significant work in RPC_Stats_Measurement addition if
no errors have occurred.

#### Problem Description
rpc-stats.h
rpc-stats.cc

#### Code Diff
```diff
- struct RPC_Stats_Measurement {
-   ...
-   double errors[RPC::NUM_ERRORS];
+ struct RPC_Stats_Measurement {
+   ...
+   double get_errors(int index) const { return errors[index]; }
+   void set_errors(int index, double value) {
+     errors[index] = value;
+     any_errors_set = true;
+   }
+  private:
+   ...
+   // We make this private so that we can keep track of whether any of
+   // these values have been set to non-zero values.
+   double errors[RPC::NUM_ERRORS];
+   bool any_errors_set;  // True iff any of the errors[i] values are non-zero
```

### Do array lookup on first byte of string to often avoid
fingerprinting full string.

#### Problem Description
soft-tokens-helper.cc
soft-tokens-helper.h
soft-tokens-helper.cc

#### Code Diff
```diff
- bool SoftTokensHelper::IsSoftToken(const StringPiece& token) const {
-   return soft_tokens_.find(Fingerprint(token.data(), token.size())) !=
-       soft_tokens_.end();
- }
+ class SoftTokensHelper {
+  ...
+  private:
+   ...
+   // Since soft tokens are mostly punctuation-related, for performance
+   // purposes, we keep an array filter_.  filter_[i] is true iff any
+   // of the soft tokens start with the byte value 'i'.  This avoids
+   // fingerprinting a term in the common case, since we can just do an array
+   // lookup based on the first byte, and if filter_[b] is false, then
+   // we can return false immediately.
+   bool          filter_[256];
+   ...
+ };
+ 
+ inline bool SoftTokensHelper::IsSoftToken(const StringPiece& token) const {
+   if (token.size() >= 1) {
+     char first_char = token.data()[0];
+     if (!filter_[first_char]) {
+       return false;
+     }
+   }
+   return IsSoftTokenFallback(token);
+ }
```

### Precompute expensive information once

### Precompute a TensorFlow graph execution node property
that allows us to quickly rule out certain unusual cases.

#### Problem Description
executor.cc

#### Code Diff
```diff
- struct NodeItem {
-   ...
-   bool kernel_is_expensive = false;  // True iff kernel->IsExpensive()
-   bool kernel_is_async = false;      // True iff kernel->AsAsync() != nullptr
-   bool is_merge = false;             // True iff IsMerge(node)
-   ...
-   if (IsEnter(node)) {
-   ...
-   } else if (IsExit(node)) {
-   ...
-   } else if (IsNextIteration(node)) {
-   ...
-   } else {
-     // Normal path for most nodes
-     ...
-   }
+ struct NodeItem {
+   ...
+   bool kernel_is_expensive : 1;  // True iff kernel->IsExpensive()
+   bool kernel_is_async : 1;      // True iff kernel->AsAsync() != nullptr
+   bool is_merge : 1;             // True iff IsMerge(node)
+   bool is_enter : 1;             // True iff IsEnter(node)
+   bool is_exit : 1;              // True iff IsExit(node)
+   bool is_control_trigger : 1;   // True iff IsControlTrigger(node)
+   bool is_sink : 1;              // True iff IsSink(node)
+   // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
+   bool is_enter_exit_or_next_iter : 1;
+   ...
+   if (!item->is_enter_exit_or_next_iter) {
+     // Fast path for nodes types that don't need special handling
+     DCHECK_EQ(input_frame, output_frame);
+     ...
+   } else if (item->is_enter) {
+   ...
+   } else if (item->is_exit) {
+   ...
+   } else {
+     DCHECK(IsNextIteration(node));
+     ...
+   }
```

### Precompute 256 element array and use during trigram
initialization.

#### Problem Description
byte_trigram_classifier.cc

#### Code Diff
```diff
- void ByteTrigramClassifier::VerifyModel(void) const {
-   ProbT class_sums[num_classes_];
-   for (int cls = 0; cls < num_classes_; cls++) {
-     class_sums[cls] = 0;
-   }
-   for (ByteNgramId id = 0; id < trigrams_.num_trigrams(); id++) {
-     for (int cls = 0; cls < num_classes_; ++cls) {
-       class_sums[cls] += Prob(trigram_probs_[id].log_probs[cls]);
-     }
-   }
-   ...
- }
+ void ByteTrigramClassifier::VerifyModel(void) const {
+   CHECK_EQ(sizeof(ByteLogProbT), 1);
+   ProbT fast_prob[256];
+   for (int b = 0; b < 256; b++) {
+     fast_prob[b] = Prob(static_cast<ByteLogProbT>(b));
+   }
+ 
+   ProbT class_sums[num_classes_];
+   for (int cls = 0; cls < num_classes_; cls++) {
+     class_sums[cls] = 0;
+   }
+   for (ByteNgramId id = 0; id < trigrams_.num_trigrams(); id++) {
+     for (int cls = 0; cls < num_classes_; ++cls) {
+       class_sums[cls] += fast_prob[trigram_probs_[id].log_probs[cls]];
+     }
+   }
+   ...
+ }
```

General advice: check for malformed inputs at module boundaries instead of
repeating checks internally.

### Move expensive computations outside loops

### Move bounds computation outside loop.

#### Problem Description
literal_linearizer.cc

#### Code Diff
```diff
- for (int64 i = 0; i < src_shape.dimensions(dimension_numbers.front());
-      ++i) {
+ int64 dim_front = src_shape.dimensions(dimension_numbers.front());
+ const uint8* src_buffer_data = src_buffer.data();
+ uint8* dst_buffer_data = dst_buffer.data();
+ for (int64 i = 0; i < dim_front; ++i) {
```

### Defer expensive computation

### Defer GetSubSharding call until needed, which reduces 43
seconds of CPU time to 2 seconds.

#### Problem Description
sharding_propagation.cc

#### Code Diff
```diff
- HloSharding alternative_sub_sharding =
-     user.sharding().GetSubSharding(user.shape(), {i});
- if (user.operand(i) == &instruction &&
-     hlo_sharding_util::IsShardingMoreSpecific(alternative_sub_sharding,
-                                               sub_sharding)) {
-   sub_sharding = alternative_sub_sharding;
- }
+ if (user.operand(i) == &instruction) {
+   // Only evaluate GetSubSharding if this operand is of interest,
+   // as it is relatively expensive.
+   HloSharding alternative_sub_sharding =
+       user.sharding().GetSubSharding(user.shape(), {i});
+   if (hlo_sharding_util::IsShardingMoreSpecific(
+           alternative_sub_sharding, sub_sharding)) {
+     sub_sharding = alternative_sub_sharding;
+   }
+ }
```

### Don't update stats eagerly; compute them on demand.

### 不要急于更新统计数据；按需计算它们。

#### Problem Description
Do not update stats on the very frequent allocation/deallocation calls. Instead,
compute stats on demand when the much less frequently called Stats() method is
invoked.

### Preallocate 10 nodes not 200 for query handling in Google's
web server.

#### Problem Description
A simple change that reduced web server's CPU usage by 7.5%.
querytree.h

#### Code Diff
```diff
- static const int kInitParseTreeSize = 200;   // initial size of querynode pool
+ static const int kInitParseTreeSize = 10;   // initial size of querynode pool
```

### Change search order for 19% throughput improvement.

### 更改搜索顺序以提高19%的吞吐量。

#### Problem Description
An old search system (circa 2000) had two tiers: one contained a full-text
index, and the other tier contained just the index for the title and anchor
terms. We used to search the smaller title/anchor tier first.
Counter-intuitively, we found that it is cheaper to search the larger full-text
index tier first since if we reach the end of the full-text tier, we can
entirely skip searching the title/anchor tier (a subset of the full-text tier).
This happened reasonably often and allowed us to reduce the average number of
disk seeks to process a query.
See discussion of title and anchor text handling in
The Anatomy of a Large-Scale Hypertextual Web Search Engine
for background information.

#### 问题描述
一个旧的搜索系统（大约在2000年）有两个层级：一个包含全文
索引，另一个层级仅包含标题和锚文本的索引。我们过去常常先搜索较小的标题/锚文本层级。
与直觉相反，我们发现先搜索较大的全文
索引层级更便宜，因为如果我们到达了全文层级的末尾，我们就可以
完全跳过搜索标题/锚文本层级（全文层级的一个子集）。
这种情况经常发生，使我们能够减少处理查询所需的平均磁盘寻道次数。
有关标题和锚文本处理的讨论，请参阅
《大规模超文本网络搜索引擎的剖析》。

### Specialize code

A particular performance-sensitive call-site may not need the full generality
provided by a general-purpose library. Consider writing specialized code in such
cases instead of calling the general-purpose code if it provides a performance
improvement.

### Custom printing code for Histogram class is 4x as fast as
sprintf.

#### Problem Description
This code is performance sensitive because it is invoked when monitoring systems
gather statistics from various servers.
histogram_export.cc

#### Code Diff
```diff
- void Histogram::PopulateBuckets(const string &prefix,
-                                 expvar::MapProto *const var) const {
-                                 ...
-   for (int i = min_bucket; i <= max_bucket; ++i) {
-     const double count = BucketCount(i);
-     if (!export_empty_buckets && count == 0.0) continue;
-     acc += count;
-     // The label format of exported buckets for discrete histograms
-     // specifies an inclusive upper bound, which is the same as in
-     // the original Histogram implementation.  This format is not
-     // applicable to non-discrete histograms, so a half-open interval
-     // is used for them, with "_" instead of "-" as a separator to
-     // make possible to distinguish the formats.
-     string key =
-         options_.export_cumulative_counts() ?
-             StringPrintf("%.12g", boundaries_->BucketLimit(i)) :
-         options_.discrete() ?
-             StringPrintf("%.0f-%.0f",
-                          ceil(boundaries_->BucketStart(i)),
-                          ceil(boundaries_->BucketLimit(i)) - 1.0) :
-             StringPrintf("%.12g_%.12g",
-                          boundaries_->BucketStart(i),
-                          boundaries_->BucketLimit(i));
-     EscapeMapKey(&key);
-     const double value = options_.export_cumulative_counts() ? acc : count;
-     expvar::AddMapFloat(StrCat(prefix,
-                                options_.export_bucket_key_prefix(),
-                                key),
-                         value * count_mult,
-                         var);
-   }
+ // Format "val" according to format.  If "need_escape" is true, then the
+ // format can produce output with a '.' in it, and the result will be escaped.
+ // If "need_escape" is false, then the caller guarantees that format is
+ // such that the resulting number will not have any '.' characters and
+ // therefore we can avoid calling EscapeKey.
+ // The function is free to use "*scratch" for scratch space if necessary,
+ // and the resulting StringPiece may point into "*scratch".
+ static StringPiece FormatNumber(const char* format,
+                                 bool need_escape,
+                                 double val, string* scratch) {
+   // This routine is specialized to work with only a limited number of formats
+   DCHECK(StringPiece(format) == "%.0f" || StringPiece(format) == "%.12g");
+ 
+   scratch->clear();
+   if (val == trunc(val) && val >= kint32min && val <= kint32max) {
+     // An integer for which we can just use StrAppend
+     StrAppend(scratch, static_cast<int32>(val));
+     return StringPiece(*scratch);
+   } else if (isinf(val)) {
+     // Infinity, represent as just 'inf'.
+     return StringPiece("inf", 3);
+   } else {
+     // Format according to "format", and possibly escape.
+     StringAppendF(scratch, format, val);
+     if (need_escape) {
+       EscapeMapKey(scratch);
+     } else {
+       DCHECK(!StringPiece(*scratch).contains("."));
+     }
+     return StringPiece(*scratch);
+   }
+ }
+ ...
+ void Histogram::PopulateBuckets(const string &prefix,
+                                 expvar::MapProto *const var) const {
+                                 ...
+   const string full_key_prefix = StrCat(prefix,
+                                         options_.export_bucket_key_prefix());
+   string key = full_key_prefix;  // Keys will start with "full_key_prefix".
+   string start_scratch;
+   string limit_scratch;
+   const bool cumul_counts = options_.export_cumulative_counts();
+   const bool discrete = options_.discrete();
+   for (int i = min_bucket; i <= max_bucket; ++i) {
+     const double count = BucketCount(i);
+     if (!export_empty_buckets && count == 0.0) continue;
+     acc += count;
+     // The label format of exported buckets for discrete histograms
+     // specifies an inclusive upper bound, which is the same as in
+     // the original Histogram implementation.  This format is not
+     // applicable to non-discrete histograms, so a half-open interval
+     // is used for them, with "_" instead of "-" as a separator to
+     // make possible to distinguish the formats.
+     key.resize(full_key_prefix.size());  // Start with full_key_prefix.
+     DCHECK_EQ(key, full_key_prefix);
+ 
+     const double limit = boundaries_->BucketLimit(i);
+     if (cumul_counts) {
+       StrAppend(&key, FormatNumber("%.12g", true, limit, &limit_scratch));
+     } else {
+       const double start = boundaries_->BucketStart(i);
+       if (discrete) {
+         StrAppend(&key,
+                   FormatNumber("%.0f", false, ceil(start), &start_scratch),
+                   "-",
+                   FormatNumber("%.0f", false, ceil(limit) - 1.0,
+                                &limit_scratch));
+       } else {
+         StrAppend(&key,
+                   FormatNumber("%.12g", true, start, &start_scratch),
+                   "_",
+                   FormatNumber("%.12g", true, limit, &limit_scratch));
+       }
+     }
+     const double value = cumul_counts ? acc : count;
+ 
+     // Add to map var
+     expvar::AddMapFloat(key, value * count_mult, var);
+   }
+ }
```

### Add specializations for VLOG(1), VLOG(2), … for speed and
smaller code size.

#### Problem Description
VLOG is a heavily used macro throughout the code base. This change avoids
passing an extra integer constant at nearly every call site (if the log level is
constant at the call site, as it almost always is, as in VLOG(1) << ...),
which saves code space.
vlog_is_on.h
vlog_is_on.cc

#### Code Diff
```diff
- class VLogSite final {
-  public:
-   ...
-   bool IsEnabled(int level) {
-     int stale_v = v_.load(std::memory_order_relaxed);
-     if (ABSL_PREDICT_TRUE(level > stale_v)) {
-       return false;
-     }
- 
-     // We put everything other than the fast path, i.e. vlogging is initialized
-     // but not on, behind an out-of-line function to reduce code size.
-     return SlowIsEnabled(stale_v, level);
-   }
-   ...
-  private:
-   ...
-   ABSL_ATTRIBUTE_NOINLINE
-   bool SlowIsEnabled(int stale_v, int level);
-   ...
- };
+ class VLogSite final {
+  public:
+   ...
+   bool IsEnabled(int level) {
+     int stale_v = v_.load(std::memory_order_relaxed);
+     if (ABSL_PREDICT_TRUE(level > stale_v)) {
+       return false;
+     }
+ 
+     // We put everything other than the fast path, i.e. vlogging is initialized
+     // but not on, behind an out-of-line function to reduce code size.
+     // "level" is almost always a call-site constant, so we can save a bit
+     // of code space by special-casing for levels 1, 2, and 3.
+ #if defined(__has_builtin) && __has_builtin(__builtin_constant_p)
+     if (__builtin_constant_p(level)) {
+       if (level == 0) return SlowIsEnabled0(stale_v);
+       if (level == 1) return SlowIsEnabled1(stale_v);
+       if (level == 2) return SlowIsEnabled2(stale_v);
+       if (level == 3) return SlowIsEnabled3(stale_v);
+       if (level == 4) return SlowIsEnabled4(stale_v);
+       if (level == 5) return SlowIsEnabled5(stale_v);
+     }
+ #endif
+     return SlowIsEnabled(stale_v, level);
+     ...
+  private:
+   ...
+   ABSL_ATTRIBUTE_NOINLINE
+   bool SlowIsEnabled(int stale_v, int level);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled0(int stale_v);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled1(int stale_v);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled2(int stale_v);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled3(int stale_v);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled4(int stale_v);
+   ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled5(int stale_v);
+   ...
+ };
```

### Replace RE2 call with a simple prefix match when possible.

#### Problem Description
read_matcher.cc
read_matcher.cc

#### Code Diff
```diff
- enum MatchItemType {
-   MATCH_TYPE_INVALID,
-   MATCH_TYPE_RANGE,
-   MATCH_TYPE_EXACT,
-   MATCH_TYPE_REGEXP,
- };
+ enum MatchItemType {
+   MATCH_TYPE_INVALID,
+   MATCH_TYPE_RANGE,
+   MATCH_TYPE_EXACT,
+   MATCH_TYPE_REGEXP,
+   MATCH_TYPE_PREFIX,   // Special type for regexp ".*"
+ };
```

### Use StrCat rather than StringPrintf to format IP
addresses.

#### Problem Description
ipaddress.cc

#### Code Diff
```diff
- string IPAddress::ToString() const {
-   char buf[INET6_ADDRSTRLEN];
- 
-   switch (address_family_) {
-     case AF_INET:
-       CHECK(inet_ntop(AF_INET, &addr_.addr4, buf, INET6_ADDRSTRLEN) != NULL);
-       return buf;
-     case AF_INET6:
-       CHECK(inet_ntop(AF_INET6, &addr_.addr6, buf, INET6_ADDRSTRLEN) != NULL);
-       return buf;
-     case AF_UNSPEC:
-       LOG(DFATAL) << "Calling ToString() on an empty IPAddress";
-       return "";
-     default:
-       LOG(FATAL) << "Unknown address family " << address_family_;
-   }
- }
- ...
- string IPAddressToURIString(const IPAddress& ip) {
-   switch (ip.address_family()) {
-     case AF_INET6:
-       return StringPrintf("[%s]", ip.ToString().c_str());
-     default:
-       return ip.ToString();
-   }
- }
- ...
- string SocketAddress::ToString() const {
-   return IPAddressToURIString(host_) + StringPrintf(":%u", port_);
- }
+ string IPAddress::ToString() const {
+   char buf[INET6_ADDRSTRLEN];
+ 
+   switch (address_family_) {
+     case AF_INET: {
+       uint32 addr = gntohl(addr_.addr4.s_addr);
+       int a1 = static_cast<int>((addr >> 24) & 0xff);
+       int a2 = static_cast<int>((addr >> 16) & 0xff);
+       int a3 = static_cast<int>((addr >> 8) & 0xff);
+       int a4 = static_cast<int>(addr & 0xff);
+       return StrCat(a1, ".", a2, ".", a3, ".", a4);
+     }
+     case AF_INET6:
+       CHECK(inet_ntop(AF_INET6, &addr_.addr6, buf, INET6_ADDRSTRLEN) != NULL);
+       return buf;
+     case AF_UNSPEC:
+       LOG(DFATAL) << "Calling ToString() on an empty IPAddress";
+       return "";
+     default:
+       LOG(FATAL) << "Unknown address family " << address_family_;
+   }
+ }
+ ...
+ string IPAddressToURIString(const IPAddress& ip) {
+   switch (ip.address_family()) {
+     case AF_INET6:
+       return StrCat("[", ip.ToString(), "]");
+     default:
+       return ip.ToString();
+   }
+ }
+ ...
+ string SocketAddress::ToString() const {
+   return StrCat(IPAddressToURIString(host_), ":", port_);
+ }
```

### Use caching to avoid repeated work

### Cache based on precomputed fingerprint of large
serialized proto.

#### Problem Description
dp_ops.cc

#### Code Diff
```diff
- InputOutputMappingProto mapping_proto;
- PLAQUE_OP_REQUIRES(
-     mapping_proto.ParseFromStringPiece(GetAttrMappingProto(state)),
-     absl::InternalError("Failed to parse InputOutputMappingProto"));
- ParseMapping(mapping_proto);
+ uint64 mapping_proto_fp = GetAttrMappingProtoFp(state);
+ {
+   absl::MutexLock l(&fp_to_iometa_mu);
+   if (fp_to_iometa == nullptr) {
+     fp_to_iometa =
+         new absl::flat_hash_map<uint64, std::unique_ptr<ProgramIOMetadata>>;
+   }
+   auto it = fp_to_iometa->find(mapping_proto_fp);
+   if (it != fp_to_iometa->end()) {
+     io_metadata_ = it->second.get();
+   } else {
+     auto serial_proto = GetAttrMappingProto(state);
+     DCHECK_EQ(mapping_proto_fp, Fingerprint(serial_proto));
+     InputOutputMappingProto mapping_proto;
+     PLAQUE_OP_REQUIRES(
+         mapping_proto.ParseFromStringPiece(GetAttrMappingProto(state)),
+         absl::InternalError("Failed to parse InputOutputMappingProto"));
+     auto io_meta = ParseMapping(mapping_proto);
+     io_metadata_ = io_meta.get();
+     (*fp_to_iometa)[mapping_proto_fp] = std::move(io_meta);
+   }
+ }
```

### Make the compiler’s job easier

The compiler may have trouble optimizing through layers of abstractions because
it must make conservative assumptions about the overall behavior of the code, or
may not make the right speed vs. size tradeoffs. The application programmer will
often know more about the behavior of the system and can aid the compiler by
rewriting the code to operate at a lower level. However, only do this when
profiles show an issue since compilers will often get things right on their own.
Looking at the generated assembly code for performance critical routines can
help you understand if the compiler is “getting it right”. Pprof provides a very
helpful display of source code interleaved with disassembly
and annotated with performance data.

Some techniques that may be useful:

<ol>
<li>Avoid functions calls in hot functions (allows the compiler to avoid frame
setup costs).</li>
<li>Move slow-path code into a separate tail-called function.</li>
<li>Copy small amounts of data into local variables before heavy use. This can
let the compiler assume there is no aliasing with other data, which may
improve auto-vectorization and register allocation.</li>
<li>Hand-unroll very hot loops.</li>
</ol>

### Speed up ShapeUtil::ForEachState by replacing absl::Span
with raw pointers to the underlying arrays.

#### Problem Description
shape_util.h

#### Code Diff
```diff
- struct ForEachState {
-   ForEachState(const Shape& s, absl::Span<const int64_t> b,
-                absl::Span<const int64_t> c, absl::Span<const int64_t> i);
-   ~ForEachState();
- 
-   const Shape& shape;
-   const absl::Span<const int64_t> base;
-   const absl::Span<const int64_t> count;
-   const absl::Span<const int64_t> incr;
+ struct ForEachState {
+   ForEachState(const Shape& s, absl::Span<const int64_t> b,
+                absl::Span<const int64_t> c, absl::Span<const int64_t> i);
+   inline ~ForEachState() = default;
+ 
+   const Shape& shape;
+   // Pointers to arrays of the passed-in spans
+   const int64_t* const base;
+   const int64_t* const count;
+   const int64_t* const incr;
```

### Hand unroll
cyclic
redundancy check (CRC) computation loop.

#### Problem Description
crc.cc

#### Code Diff
```diff
- void CRC32::Extend(uint64 *lo, uint64 *hi, const void *bytes, size_t length)
-                       const {
-                       ...
-   // Process bytes 4 at a time
-   while ((p + 4) <= e) {
-     uint32 c = l ^ WORD(p);
-     p += 4;
-     l = this->table3_[c & 0xff] ^
-         this->table2_[(c >> 8) & 0xff] ^
-         this->table1_[(c >> 16) & 0xff] ^
-         this->table0_[c >> 24];
-   }
- 
-   // Process the last few bytes
-   while (p != e) {
-     int c = (l & 0xff) ^ *p++;
-     l = this->table0_[c] ^ (l >> 8);
-   }
-   *lo = l;
- }
+ void CRC32::Extend(uint64 *lo, uint64 *hi, const void *bytes, size_t length)
+                       const {
+                       ...
+ #define STEP {                                  \
+     uint32 c = l ^ WORD(p);                     \
+     p += 4;                                     \
+     l = this->table3_[c & 0xff] ^               \
+         this->table2_[(c >> 8) & 0xff] ^        \
+         this->table1_[(c >> 16) & 0xff] ^       \
+         this->table0_[c >> 24];                 \
+ }
+ 
+   // Process bytes 16 at a time
+   while ((e-p) >= 16) {
+     STEP;
+     STEP;
+     STEP;
+     STEP;
+   }
+ 
+   // Process bytes 4 at a time
+   while ((p + 4) <= e) {
+     STEP;
+   }
+ #undef STEP
+ 
+   // Process the last few bytes
+   while (p != e) {
+     int c = (l & 0xff) ^ *p++;
+     l = this->table0_[c] ^ (l >> 8);
+   }
+   *lo = l;
+ }
```

### Handle four characters at a time when parsing Spanner
keys.

#### Problem Description
key.cc

#### Code Diff
```diff
- void Key::InitSeps(const char* start) {
-   const char* base = &rep_[0];
-   const char* limit = base + rep_.size();
-   const char* s = start;
- 
-   DCHECK_GE(s, base);
-   DCHECK_LT(s, limit);
- 
-   for (int i = 0; i < 3; i++) {
-     s = (const char*)memchr(s, '#', limit - s);
-     DCHECK(s != NULL);
-     seps_[i] = s - base;
-     s++;
-   }
- }
+ inline const char* ScanBackwardsForSep(const char* base, const char* p) {
+   while (p >= base + 4) {
+     if (p[0] == '#') return p;
+     if (p[-1] == '#') return p-1;
+     if (p[-2] == '#') return p-2;
+     if (p[-3] == '#') return p-3;
+     p -= 4;
+   }
+   while (p >= base && *p != '#') p--;
+   return p;
+ }
+ 
+ void Key::InitSeps(const char* start) {
+   const char* base = &rep_[0];
+   const char* limit = base + rep_.size();
+   const char* s = start;
+ 
+   DCHECK_GE(s, base);
+   DCHECK_LT(s, limit);
+ 
+   // We go backwards from the end of the string, rather than forwards,
+   // since the directory name might be long and definitely doesn't contain
+   // any '#' characters.
+   const char* p = ScanBackwardsForSep(s, limit - 1);
+   DCHECK(*p == '#');
+   seps_[2] = p - base;
+   p--;
+ 
+   p = ScanBackwardsForSep(s, p);
+   DCHECK(*p == '#');
+   seps_[1] = p - base;
+   p--;
+ 
+   p = ScanBackwardsForSep(s, p);
+   DCHECK(*p == '#');
+   seps_[0] = p - base;
+ }
```

### Avoid frame setup costs by converting ABSL_LOG(FATAL) to
ABSL_DCHECK(false).

#### Problem Description
arena_cleanup.h

#### Code Diff
```diff
- inline ABSL_ATTRIBUTE_ALWAYS_INLINE size_t Size(Tag tag) {
-   if (!EnableSpecializedTags()) return sizeof(DynamicNode);
- 
-   switch (tag) {
-     case Tag::kDynamic:
-       return sizeof(DynamicNode);
-     case Tag::kString:
-       return sizeof(TaggedNode);
-     case Tag::kCord:
-       return sizeof(TaggedNode);
-     default:
-       ABSL_LOG(FATAL) << "Corrupted cleanup tag: " << static_cast<int>(tag);
-       return sizeof(DynamicNode);
-   }
- }
+ inline ABSL_ATTRIBUTE_ALWAYS_INLINE size_t Size(Tag tag) {
+   if (!EnableSpecializedTags()) return sizeof(DynamicNode);
+ 
+   switch (tag) {
+     case Tag::kDynamic:
+       return sizeof(DynamicNode);
+     case Tag::kString:
+       return sizeof(TaggedNode);
+     case Tag::kCord:
+       return sizeof(TaggedNode);
+     default:
+       ABSL_DCHECK(false) << "Corrupted cleanup tag: " << static_cast<int>(tag);
+       return sizeof(DynamicNode);
+   }
+ }
```

### Reduce stats collection costs

Balance the utility of stats and other behavioral information about a system
against the cost of maintaining that information. The extra information can
often help people to understand and improve high-level behavior, but can also be
costly to maintain.

Stats that are not useful can be dropped altogether.

### Stop maintaining expensive stats about number of alarms and
closures in SelectServer.

#### Problem Description
Part of changes that reduce time for setting an alarm from 771 ns to 271 ns.
selectserver.h
/selectserver.cc
/selectserver.cc

#### Code Diff
```diff
- class SelectServer {
-  public:
-  ...
-  protected:
-   ...
-   scoped_ptr<MinuteTenMinuteHourStat> num_alarms_stat_;
-   ...
-   scoped_ptr<MinuteTenMinuteHourStat> num_closures_stat_;
-   ...
- };
+ // Selectserver class
+ class SelectServer {
+  ...
+  protected:
+  ...
+ };
```

Often, stats or other properties can be maintained for a sample of the elements
handled by the system (e.g., RPC requests, input records, users). Many
subsystems use this approach (tcmalloc allocation tracking, /requestz status
pages, Dapper samples).

When sampling, consider reducing the sampling rate when appropriate.

### Maintain stats for just a sample of doc info requests.

#### Problem Description
Sampling allows us to avoid touching 39 histograms and MinuteTenMinuteHour stats
for most requests.
generic-leaf-stats.cc

### Reduce sampling rate and make faster sampling decisions.

#### Problem Description
This change reduces the sampling rate from 1 in 10 to 1 in 32. Furthermore, we
now keep execution time stats just for the sampled events and speed up sampling
decisions by using a power of two modulus. This code is called on every packet
in the Google Meet video conferencing system and needed performance work to keep
up with capacity demands during the first part of the COVID outbreak as users
rapidly migrated to doing more online meetings.
packet_executor.cc
packet_executor.cc
Benchmark results:

#### Code Diff
```diff
- class ScopedPerformanceMeasurement {
-  public:
-   explicit ScopedPerformanceMeasurement(PacketExecutor* packet_executor)
-       : packet_executor_(packet_executor),
-         tracer_(packet_executor->packet_executor_trace_threshold_,
-                 kClosureTraceName) {
-     // ThreadCPUUsage is an expensive call. At the time of writing,
-     // it takes over 400ns, or roughly 30 times slower than absl::Now,
-     // so we sample only 10% of closures to keep the cost down.
-     if (packet_executor->closures_executed_ % 10 == 0) {
-       thread_cpu_usage_start_ = base::ThreadCPUUsage();
-     }
- 
-     // Sample start time after potentially making the above expensive call,
-     // so as not to pollute wall time measurements.
-     run_start_time_ = absl::Now();
-   }
- 
-   ~ScopedPerformanceMeasurement() {
+ ScopedPerformanceMeasurement::ScopedPerformanceMeasurement(
+     PacketExecutor* packet_executor)
+     : packet_executor_(packet_executor),
+       tracer_(packet_executor->packet_executor_trace_threshold_,
+               kClosureTraceName) {
+   // ThreadCPUUsage is an expensive call. At the time of writing,
+   // it takes over 400ns, or roughly 30 times slower than absl::Now,
+   // so we sample only 1 in 32 closures to keep the cost down.
+   if (packet_executor->closures_executed_ % 32 == 0) {
+     thread_cpu_usage_start_ = base::ThreadCPUUsage();
+   }
+ 
+   // Sample start time after potentially making the above expensive call,
+   // so as not to pollute wall time measurements.
+   run_start_time_ = absl::Now();
+ }
```

### Avoid logging on hot code paths

### 避免在热代码路径上记录日志

Logging statements can be costly, even if the logging-level for the statement
doesn’t actually log anything. E.g., ABSL_VLOG’s implementation requires at
least a load and a comparison, which may be a problem in hot code paths. In
addition, the presence of the logging code may inhibit compiler optimizations.
Consider dropping logging entirely from hot code paths.

日志语句可能会很昂贵，即使语句的日志级别
实际上没有记录任何东西。例如，ABSL_VLOG的实现至少需要
一次加载和一次比较，这在热代码路径中可能会成为问题。此外，日志代码的存在可能会抑制编译器优化。
考虑从热代码路径中完全删除日志记录。

### Remove logging from guts of memory allocator.

#### Problem Description
This was a small part of a larger change.
gpu_bfc_allocator.cc

### Precompute whether or not logging is enabled outside a
nested loop.

#### Problem Description
image_similarity.cc

#### Code Diff
```diff
- for (int j = 0; j < output_subimage_size_y; j++) {
-   int j1 = j - rad + output_to_integral_subimage_y;
-   int j2 = j1 + 2 * rad + 1;
-   // Create a pointer for this row's output, taking into account the offset
-   // to the full image.
-   double *image_diff_ptr = &(*image_diff)(j + min_j, min_i);
- 
-   for (int i = 0; i < output_subimage_size_x; i++) {
-     ...
-     if (VLOG_IS_ON(3)) {
-     ...
-     }
-     ...
-   }
- }
+ const bool vlog_3 = DEBUG_MODE ? VLOG_IS_ON(3) : false;
+ 
+ for (int j = 0; j < output_subimage_size_y; j++) {
+   int j1 = j - rad + output_to_integral_subimage_y;
+   int j2 = j1 + 2 * rad + 1;
+   // Create a pointer for this row's output, taking into account the offset
+   // to the full image.
+   double *image_diff_ptr = &(*image_diff)(j + min_j, min_i);
+ 
+   for (int i = 0; i < output_subimage_size_x; i++) {
+     ...
+     if (vlog_3) {
+     ...
+     }
+   }
+ }
```

### Precompute whether logging is enabled and use the result
in helper routines.

#### Problem Description
periodic_call.cc

#### Code Diff
```diff
- VLOG(1) << Logid()
-           << "MaybeScheduleAlarmAtNextTick. Time until next real time: "
-           << time_until_next_real_time;
-           ...
-   uint64 next_virtual_time_ms =
-       next_virtual_time_ms_ - num_ticks * kResolutionMs;
-   CHECK_GE(next_virtual_time_ms, 0);
-   ScheduleAlarm(now, delay, next_virtual_time_ms);
- }
- 
- void ScheduleNextAlarm(uint64 current_virtual_time_ms)
-     ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
-   if (calls_.empty()) {
-     VLOG(1) << Logid() << "No calls left, entering idle mode";
-     next_real_time_ = absl::InfiniteFuture();
-     return;
-   }
-   uint64 next_virtual_time_ms = FindNextVirtualTime(current_virtual_time_ms);
-   auto delay =
-       absl::Milliseconds(next_virtual_time_ms - current_virtual_time_ms);
-   ScheduleAlarm(GetClock().TimeNow(), delay, next_virtual_time_ms);
- }
- 
- // An alarm scheduled by this function supersedes all previously scheduled
- // alarms. This is ensured through `scheduling_sequence_number_`.
- void ScheduleAlarm(absl::Time now, absl::Duration delay,
-                    uint64 virtual_time_ms)
-     ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
-   next_real_time_ = now + delay;
-   next_virtual_time_ms_ = virtual_time_ms;
-   ++ref_count_;  // The Alarm holds a reference.
-   ++scheduling_sequence_number_;
-   VLOG(1) << Logid() << "ScheduleAlarm. Time : "
-           << absl::FormatTime("%M:%S.%E3f", now, absl::UTCTimeZone())
-           << ", delay: " << delay << ", virtual time: " << virtual_time_ms
-           << ", refs: " << ref_count_
-           << ", seq: " << scheduling_sequence_number_
-           << ", executor: " << executor_;
- 
-   executor_->AddAfter(
-       delay, new Alarm(this, virtual_time_ms, scheduling_sequence_number_));
- }
+ const bool vlog_1 = VLOG_IS_ON(1);
+ 
+   if (vlog_1) {
+     VLOG(1) << Logid()
+             << "MaybeScheduleAlarmAtNextTick. Time until next real time: "
+             << time_until_next_real_time;
+   }
+   ...
+   uint64 next_virtual_time_ms =
+       next_virtual_time_ms_ - num_ticks * kResolutionMs;
+   CHECK_GE(next_virtual_time_ms, 0);
+   ScheduleAlarm(now, delay, next_virtual_time_ms, vlog_1);
+ }
+ 
+ void ScheduleNextAlarm(uint64 current_virtual_time_ms, bool vlog_1)
+     ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
+   if (calls_.empty()) {
+     if (vlog_1) {
+       VLOG(1) << Logid() << "No calls left, entering idle mode";
+     }
+     next_real_time_ = absl::InfiniteFuture();
+     return;
+   }
+   uint64 next_virtual_time_ms = FindNextVirtualTime(current_virtual_time_ms);
+   auto delay =
+       absl::Milliseconds(next_virtual_time_ms - current_virtual_time_ms);
+   ScheduleAlarm(GetClock().TimeNow(), delay, next_virtual_time_ms, vlog_1);
+ }
+ 
+ // An alarm scheduled by this function supersedes all previously scheduled
+ // alarms. This is ensured through `scheduling_sequence_number_`.
+ void ScheduleAlarm(absl::Time now, absl::Duration delay,
+                    uint64 virtual_time_ms,
+                    bool vlog_1)
+     ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
+   next_real_time_ = now + delay;
+   next_virtual_time_ms_ = virtual_time_ms;
+   ++ref_count_;  // The Alarm holds a reference.
+   ++scheduling_sequence_number_;
+   if (vlog_1) {
+     VLOG(1) << Logid() << "ScheduleAlarm. Time : "
+             << absl::FormatTime("%M:%S.%E3f", now, absl::UTCTimeZone())
+             << ", delay: " << delay << ", virtual time: " << virtual_time_ms
+             << ", refs: " << ref_count_
+             << ", seq: " << scheduling_sequence_number_
+             << ", executor: " << executor_;
+   }
+ 
+   executor_->AddAfter(
+       delay, new Alarm(this, virtual_time_ms, scheduling_sequence_number_));
+ }
```

## Code size considerations

## 代码大小注意事项

Performance encompasses more than just runtime speed. Sometimes it is worth
considering the effects of software choices on the size of generated code. Large
code size means longer compile and link times, bloated binaries, more memory
usage, more icache pressure, and other sometimes negative effects on
microarchitectural structures like branch predictors, etc.  Thinking about these issues is especially
important when writing low-level library code that will be used in many places,
or when writing templated code that you expect will be instantiated for many
different types.

The techniques that are useful for reducing code size vary significantly across
programming languages. Here are some techniques that we have found useful for
C++ code (which can suffer from an over-use of templates and inlining).

### Trim commonly inlined code

Widely called functions combined with inlining can have a dramatic effect on
code size.

### Speed up TF_CHECK_OK.

#### Problem Description
Avoid creating Ok object, and save code space by doing complex formatting of
fatal error message out of line instead of at every call site.
status.h
status.cc

#### Code Diff
```diff
- #define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))
- #define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))
+ extern tensorflow::string* TfCheckOpHelperOutOfLine(
+     const ::tensorflow::Status& v, const char* msg);
+ inline tensorflow::string* TfCheckOpHelper(::tensorflow::Status v,
+                                            const char* msg) {
+   if (v.ok()) return nullptr;
+   return TfCheckOpHelperOutOfLine(v, msg);
+ }
+ #define TF_CHECK_OK(val)                                           \
+   while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \
+   LOG(FATAL) << *(_result)
+ #define TF_QCHECK_OK(val)                                          \
+   while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \
+   LOG(QFATAL) << *(_result)
```

### Shrink each RETURN_IF_ERROR call site by 79 bytes of
code.

### Improve performance of CHECK_GE by 4.5X and shrink code
size from 125 bytes to 77 bytes.

#### Problem Description
logging.h
logging.cc

#### Code Diff
```diff
- struct CheckOpString {
-   CheckOpString(string* str) : str_(str) { }
-   ~CheckOpString() { delete str_; }
-   operator bool() const { return str_ == NULL; }
-   string* str_;
- };
- ...
- #define DEFINE_CHECK_OP_IMPL(name, op) \
-   template <class t1, class t2> \
-   inline string* Check##name##Impl(const t1& v1, const t2& v2, \
-                                    const char* names) { \
-     if (v1 op v2) return NULL; \
-     else return MakeCheckOpString(v1, v2, names); \
-   } \
-   string* Check##name##Impl(int v1, int v2, const char* names);
- DEFINE_CHECK_OP_IMPL(EQ, ==)
- DEFINE_CHECK_OP_IMPL(NE, !=)
- DEFINE_CHECK_OP_IMPL(LE, <=)
- DEFINE_CHECK_OP_IMPL(LT, < )
- DEFINE_CHECK_OP_IMPL(GE, >=)
- DEFINE_CHECK_OP_IMPL(GT, > )
- #undef DEFINE_CHECK_OP_IMPL
+ struct CheckOpString {
+   CheckOpString(string* str) : str_(str) { }
+   // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
+   // so there's no point in cleaning up str_.
+   operator bool() const { return str_ == NULL; }
+   string* str_;
+ };
+ ...
+ extern string* MakeCheckOpStringIntInt(int v1, int v2, const char* names);
+ 
+ template<int, int>
+ string* MakeCheckOpString(const int& v1, const int& v2, const char* names) {
+   return MakeCheckOpStringIntInt(v1, v2, names);
+ }
+ ...
+ #define DEFINE_CHECK_OP_IMPL(name, op) \
+   template <class t1, class t2> \
+   inline string* Check##name##Impl(const t1& v1, const t2& v2, \
+                                    const char* names) { \
+     if (v1 op v2) return NULL; \
+     else return MakeCheckOpString(v1, v2, names); \
+   } \
+   inline string* Check##name##Impl(int v1, int v2, const char* names) { \
+     if (v1 op v2) return NULL; \
+     else return MakeCheckOpString(v1, v2, names); \
+   }
+ DEFINE_CHECK_OP_IMPL(EQ, ==)
+ DEFINE_CHECK_OP_IMPL(NE, !=)
+ DEFINE_CHECK_OP_IMPL(LE, <=)
+ DEFINE_CHECK_OP_IMPL(LT, < )
+ DEFINE_CHECK_OP_IMPL(GE, >=)
+ DEFINE_CHECK_OP_IMPL(GT, > )
+ #undef DEFINE_CHECK_OP_IMPL
```

### Inline with care

Inlining can often improve performance, but sometimes it can increase code size
without a corresponding performance payoff (and in some case even a performance
loss due to increased instruction cache pressure).

### Reduce inlining in TensorFlow.

#### Problem Description
The change stops inlining many non-performance-sensitive functions (e.g., error
paths and op registration code). Furthermore, slow paths of some
performance-sensitive functions are moved into non-inlined functions.
These changes reduces the size of tensorflow symbols in a typical binary by
12.2% (8814545 bytes down to 7740233 bytes)

### Protocol buffer library change. Avoid expensive inlined
code space for encoding message length for messages ≥ 128 bytes and instead
do a procedure call to a shared out-of-line routine.

#### Problem Description
Not only makes important large binaries smaller but also faster.
Bytes of generated code per line of a heavily inlined routine in one large
binary. First number represents the total bytes generated for a particular
source line including all locations where that code has been inlined.
Before:
The new codesize output with this change looks like:
coded_stream.h
coded_stream.cc

#### Code Diff
```diff
+ class PROTOBUF_EXPORT CodedOutputStream {
+   ...
+   // Like WriteVarint32()  but writing directly to the target array, and with the
+   // less common-case paths being out of line rather than inlined.
+   static uint8* WriteVarint32ToArrayOutOfLine(uint32 value, uint8* target);
+   ...
+ };
+ ...
+ inline uint8* CodedOutputStream::WriteVarint32ToArrayOutOfLine(uint32 value,
+                                                                uint8* target) {
+   target[0] = static_cast<uint8>(value);
+   if (value < 0x80) {
+     return target + 1;
+   } else {
+     return WriteVarint32ToArrayOutOfLineHelper(value, target);
+   }
+ }
```

### Reduce absl::flat_hash_set and absl::flat_hash_map code
size.

#### Problem Description
Reduces sizes of some large binaries by ~0.5%.

### Do not inline string allocation and deallocation when not
using protobuf arenas.

#### Problem Description
public/arenastring.h
internal/arenastring.cc

#### Code Diff
```diff
- if (IsDefault(default_value)) {
-     std::string* new_string = new std::string();
-     tagged_ptr_.Set(new_string);
-     return new_string;
-   } else {
-     return UnsafeMutablePointer();
-   }
- }
+ if (IsDefault(default_value)) {
+     return SetAndReturnNewString();
+   } else {
+     return UnsafeMutablePointer();
+   }
+ }
```

### Avoid inlining some routines. Create variants of routines
that take 'const char*' rather than 'const std::string&' to avoid std::string
construction code at every call site.

#### Problem Description
op.h

#### Code Diff
```diff
- class OpDefBuilderWrapper {
-  public:
-   explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
-   OpDefBuilderWrapper& Attr(std::string spec) {
-     builder_.Attr(std::move(spec));
-     return *this;
-   }
-   OpDefBuilderWrapper& Input(std::string spec) {
-     builder_.Input(std::move(spec));
-     return *this;
-   }
-   OpDefBuilderWrapper& Output(std::string spec) {
-     builder_.Output(std::move(spec));
-     return *this;
-   }
+ class OpDefBuilderWrapper {
+  public:
+   explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
+   OpDefBuilderWrapper& Attr(std::string spec) {
+     builder_.Attr(std::move(spec));
+     return *this;
+   }
+   OpDefBuilderWrapper& Attr(const char* spec) TF_ATTRIBUTE_NOINLINE {
+     return Attr(std::string(spec));
+   }
+   OpDefBuilderWrapper& Input(std::string spec) {
+     builder_.Input(std::move(spec));
+     return *this;
+   }
+   OpDefBuilderWrapper& Input(const char* spec) TF_ATTRIBUTE_NOINLINE {
+     return Input(std::string(spec));
+   }
+   OpDefBuilderWrapper& Output(std::string spec) {
+     builder_.Output(std::move(spec));
+     return *this;
+   }
+   OpDefBuilderWrapper& Output(const char* spec) TF_ATTRIBUTE_NOINLINE {
+     return Output(std::string(spec));
+   }
```

### Reduce template instantiations

Templated code can be duplicated for every possible combination of template
arguments when it is instantiated.

### Replace template argument with a regular argument.

#### Problem Description
Changed a large routine templated on a bool to instead take the bool as an extra
argument. (The bool was only being used once to select one of two string
constants, so a run-time check was just fine.) This reduced the # of
instantiations of the large routine from 287 to 143.
sharding_util_ops.cc

#### Code Diff
```diff
- template <bool Split>
- Status GetAndValidateAttributes(OpKernelConstruction* ctx,
-                                 std::vector<int32>& num_partitions,
-                                 int& num_slices, std::vector<int32>& paddings,
-                                 bool& has_paddings) {
-   absl::string_view num_partitions_attr_name =
-       Split ? kNumSplitsAttrName : kNumConcatsAttrName;
-       ...
-   return OkStatus();
- }
+ Status GetAndValidateAttributes(bool split, OpKernelConstruction* ctx,
+                                 std::vector<int32>& num_partitions,
+                                 int& num_slices, std::vector<int32>& paddings,
+                                 bool& has_paddings) {
+   absl::string_view num_partitions_attr_name =
+       split ? kNumSplitsAttrName : kNumConcatsAttrName;
+       ...
+   return OkStatus();
+ }
```

### Move bulky code from templated constructor to a
non-templated shared base class constructor.

#### Problem Description
Also reduce number of template instantiations from one for every combination of
<T, Device, Rank> to one for every <T> and every <Rank>.
sharding_util_ops.cc

#### Code Diff
```diff
- template <typename Device, typename T>
- class XlaSplitNDBaseOp : public OpKernel {
-  public:
-   explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
-     OP_REQUIRES_OK(
-         ctx, GetAndValidateAttributes(/*split=*/true, ctx, num_splits_,
-                                       num_slices_, paddings_, has_paddings_));
-   }
+ // Shared base class to save code space
+ class XlaSplitNDShared : public OpKernel {
+  public:
+   explicit XlaSplitNDShared(OpKernelConstruction* ctx) TF_ATTRIBUTE_NOINLINE
+       : OpKernel(ctx),
+         num_slices_(1),
+         has_paddings_(false) {
+     GetAndValidateAttributes(/*split=*/true, ctx, num_splits_, num_slices_,
+                              paddings_, has_paddings_);
+   }
```

### Reduce generated code size for absl::flat_hash_set and
absl::flat_hash_map.

### Reduce container operations

Consider the impact of map and other container operations since each call to
such and operation can produce large amounts of generated code.

### Turn many map insertion calls in a row to initialize a
hash table of emoji characters into a single bulk insert operation (188KB of
text down to 360 bytes in library linked into many binaries). 😊

#### Problem Description
textfallback_init.h

#### Code Diff
```diff
- inline void AddEmojiFallbacks(TextFallbackMap *map) {
-   (*map)[0xFE000] = &kFE000;
-   (*map)[0xFE001] = &kFE001;
-   (*map)[0xFE002] = &kFE002;
-   (*map)[0xFE003] = &kFE003;
-   (*map)[0xFE004] = &kFE004;
-   (*map)[0xFE005] = &kFE005;
-   ...
-   (*map)[0xFEE7D] = &kFEE7D;
-   (*map)[0xFEEA0] = &kFEEA0;
-   (*map)[0xFE331] = &kFE331;
- };
+ inline void AddEmojiFallbacks(TextFallbackMap *map) {
+ #define PAIR(x) {0x##x, &k##x}
+   // clang-format off
+   map->insert({
+     PAIR(FE000),
+     PAIR(FE001),
+     PAIR(FE002),
+     PAIR(FE003),
+     PAIR(FE004),
+     PAIR(FE005),
+     ...
+     PAIR(FEE7D),
+     PAIR(FEEA0),
+     PAIR(FE331)});
+   // clang-format on
+ #undef PAIR
+ };
```

### Stop inlining a heavy user of InlinedVector operations.

#### Problem Description
Moved very long routine that was being inlined from .h file to .cc (no real
performance benefit from inlining this).
reduction_ops_common.h

## Parallelization and synchronization

### Exploit parallelism

Modern machines have many cores, and they are often underutilized. Expensive
work may therefore be completed faster by parallelizing it. The most common
approach is to process different items in parallel and combine the results when
done. Typically, the items are first partitioned into batches to avoid paying
the cost of running something in parallel per item.

### Four-way parallelization improves the rate of encoding
tokens by ~3.6x.

#### Problem Description
blocked-token-coder.cc

#### Code Diff
```diff
+ MutexLock l(&encoder_threads_lock);
+ if (encoder_threads == NULL) {
+   encoder_threads = new ThreadPool(NumCPUs());
+   encoder_threads->SetStackSize(262144);
+   encoder_threads->StartWorkers();
+ }
+ encoder_threads->Add
+     (NewCallback(this,
+                  &BlockedTokenEncoder::EncodeRegionInThread,
+                  region_tokens, N, region,
+                  stats,
+                  controller_->GetClosureWithCost
+                  (NewCallback(&DummyCallback), N)));
```

### Parallelization improves decoding performance by 5x.

#### Problem Description
coding.cc

The effect on system performance should be measured carefully – if spare CPU is
not available, or if memory bandwidth is saturated, parallelization may not
help, or may even hurt.

### Amortize lock acquisition

### 摊销锁的获取

Avoid fine-grained locking to reduce the cost of Mutex operations in hot paths.
Caveat: this should only be done if the change does not increase lock
contention.

避免细粒度锁定，以减少热路径中Mutex操作的成本。
注意：只有在更改不会增加锁争用的情况下才应这样做。

### Acquire lock once to free entire tree of query nodes, rather
than reacquiring lock for every node in tree.

### 一次性获取锁以释放整个查询节点树，而不是
为树中的每个节点重新获取锁。

#### Problem Description
mustang-query.cc

#### Code Diff
```diff
- // Pool of query nodes
- ThreadSafeFreeList<MustangQuery> pool_(256);
- ...
- void MustangQuery::Release(MustangQuery* node) {
-   if (node == NULL)
-     return;
-   for (int i=0; i < node->children_->size(); ++i)
-     Release((*node->children_)[i]);
-   node->children_->clear();
-   pool_.Delete(node);
- }
+ // Pool of query nodes
+ Mutex pool_lock_;
+ FreeList<MustangQuery> pool_(256);
+ ...
+ void MustangQuery::Release(MustangQuery* node) {
+   if (node == NULL)
+     return;
+   MutexLock l(&pool_lock_);
+   ReleaseLocked(node);
+ }
+ 
+ void MustangQuery::ReleaseLocked(MustangQuery* node) {
+ #ifndef NDEBUG
+   pool_lock_.AssertHeld();
+ #endif
+   if (node == NULL)
+     return;
+   for (int i=0; i < node->children_->size(); ++i)
+     ReleaseLocked((*node->children_)[i]);
+   node->children_->clear();
+   pool_.Delete(node);
+ }
```

### Keep critical sections short

### 保持临界区简短

Avoid expensive work inside critical sections. In particular, watch out for
innocuous looking code that might be doing RPCs or accessing files.

### Reduce number of cache lines touched in critical section.

#### Problem Description
Careful data structure adjustments reduce the number of cache lines accessed
significantly and improve the performance of an ML training run by 3.3%.

### Avoid RPC while holding Mutex.

#### Problem Description
trainer.cc

#### Code Diff
```diff
- {
-   // Notify the parameter server that we are starting.
-   MutexLock l(&lock_);
-   model_ = model;
-   MaybeRecordProgress(last_global_step_);
- }
+ bool should_start_record_progress = false;
+ int64 step_for_progress = -1;
+ {
+   // Notify the parameter server that we are starting.
+   MutexLock l(&lock_);
+   model_ = model;
+   should_start_record_progress = ShouldStartRecordProgress();
+   step_for_progress = last_global_step_;
+ }
+ if (should_start_record_progress) {
+   StartRecordProgress(step_for_progress);
+ }
```

Also, be wary of expensive destructors that will run before a Mutex is unlocked
(this can often happen when the Mutex unlock is triggered by a ~MutexUnlock.)
Declaring objects with expensive destructors before MutexLock may help (assuming
it is thread-safe).

### Reduce contention by sharding

### 通过分片减少争用

Sometimes a data structure protected by a Mutex that is exhibiting high
contention can be safely split into multiple shards, each shard with its own
Mutex. (Note: this requires that there are no cross-shard invariants between the
different shards.)

### Shard a cache 16 ways which improves throughput under a
multi-threaded load by ~2x.

#### Problem Description
cache.cc

#### Code Diff
```diff
+ class ShardedLRUCache : public Cache {
+  private:
+   LRUCache shard_[kNumShards];
+   port::Mutex id_mutex_;
+   uint64_t last_id_;
+ 
+   static inline uint32_t HashSlice(const Slice& s) {
+     return Hash(s.data(), s.size(), 0);
+   }
+ 
+   static uint32_t Shard(uint32_t hash) {
+     return hash >> (32 - kNumShardBits);
+   }
+   ...
+   virtual Handle* Lookup(const Slice& key) {
+     const uint32_t hash = HashSlice(key);
+     return shard_[Shard(hash)].Lookup(key, hash);
+   }
```

### Shard spanner data structure for tracking calls.

#### Problem Description
transaction_manager.cc

#### Code Diff
```diff
- absl::MutexLock l(&active_calls_in_mu_);
- ActiveCallMap::const_iterator iter = active_calls_in_.find(m->tid());
- if (iter != active_calls_in_.end()) {
-   iter->second.ExtractElements(&m->tmp_calls_);
- }
+ ActiveCalls::LockedShard shard(active_calls_in_, m->tid());
+ const ActiveCallMap& active_calls_map = shard.active_calls_map();
+ ActiveCallMap::const_iterator iter = active_calls_map.find(m->tid());
+ if (iter != active_calls_map.end()) {
+   iter->second.ExtractElements(&m->tmp_calls_);
+ }
```

If the data structure in question is a map, consider using a concurrent hash map
implementation instead.

Be careful with the information used for shard selection. If, for example, you
use some bits of a hash value for shard selection and then those same bits end
up being used again later, the latter use may perform poorly since it sees a
skewed distribution of hash values.

### Fix information used for shard selection to prevent hash
table issues.

#### Problem Description
netmon_map_impl.h

#### Code Diff
```diff
- ConnectionBucket* GetBucket(Index index) {
-   // Rehash the hash to make sure we are not partitioning the buckets based on
-   // the original hash. If num_buckets_ is a power of 2 that would drop the
-   // entropy of the buckets.
-   size_t original_hash = absl::Hash<Index>()(index);
-   int hash = absl::Hash<size_t>()(original_hash) % num_buckets_;
-   return &buckets_[hash];
- }
+ ConnectionBucket* GetBucket(Index index) {
+   absl::Hash<std::pair<Index, size_t>> hasher{};
+   // Combine the hash with 42 to prevent shard selection using the same bits
+   // as the underlying hashtable.
+   return &buckets_[hasher({index, 42}) % num_buckets_];
+ }
```

### Shard Spanner data structure used for tracking calls.

#### Problem Description
This CL partitions the ActiveCallMap into 64 shards. Each shard is protected by
a separate mutex. A given transaction will be mapped to exactly one shard. A new
interface LockedShard(tid) is added for accessing the ActiveCallMap for a
transaction in a thread-safe manner. Example usage:
transaction_manager.cc
The results show a 69% reduction in overall wall-clock time when running the
benchmark with 8192 fibers

#### Code Diff
```diff
- {
-   absl::MutexLock l(&active_calls_in_mu_);
-   delayed_locks_timer_ring_.Add(delayed_locks_flush_time_ms, tid);
- }
+ {
+   ActiveCalls::LockedShard shard(active_calls_in_, tid);
+   shard.delayed_locks_timer_ring().Add(delayed_locks_flush_time_ms, tid);
+ }
```

### SIMD Instructions

Explore whether handling multiple items at once using
SIMD
instructions available on modern CPUs can give speedups (e.g., see
absl::flat_hash_map discussion below in Bulk Operations
section).

### Reduce false sharing

If different threads access different mutable data, consider placing the
different data items on different cache lines, e.g., in C++ using the alignas
directive. However, these directives are easy to misuse and may increase object
sizes significantly, so make sure performance measurements justify their use.

### Segregate commonly mutated fields in a different cache
line than other fields.

#### Problem Description
histogram.h

#### Code Diff
```diff
- HistogramOptions options_;
- ...
- internal::HistogramBoundaries *boundaries_;
- ...
- std::vector<double> buckets_;
- 
- double min_;             // Minimum.
- double max_;             // Maximum.
- double count_;           // Total count of occurrences.
- double sum_;             // Sum of values.
- double sum_of_squares_;  // Sum of squares of values.
- ...
- RegisterVariableExporter *exporter_;
+ HistogramOptions options_;
+   ...
+   internal::HistogramBoundaries *boundaries_;
+   ...
+   RegisterVariableExporter *exporter_;
+   ...
+   // Place the following fields in a dedicated cacheline as they are frequently
+   // mutated, so we can avoid potential false sharing.
+   ...
+ #ifndef SWIG
+   alignas(ABSL_CACHELINE_SIZE)
+ #endif
+   std::vector<double> buckets_;
+ 
+   double min_;             // Minimum.
+   double max_;             // Maximum.
+   double count_;           // Total count of occurrences.
+   double sum_;             // Sum of values.
+   double sum_of_squares_;  // Sum of squares of values.
```

### Reduce frequency of context switches

### Process small work items inline instead of on device
thread pool.

#### Problem Description
cast_op.cc

#### Code Diff
```diff
+ template <typename Device, typename Tout, typename Tin>
+ void CastMaybeInline(const Device& d, typename TTypes<Tout>::Flat o,
+                      typename TTypes<Tin>::ConstFlat i) {
+   if (o.size() * (sizeof(Tin) + sizeof(Tout)) < 16384) {
+     // Small cast on a CPU: do inline
+     o = i.template cast<Tout>();
+   } else {
+     o.device(d) = i.template cast<Tout>();
+   }
+ }
```

### Use buffered channels for pipelining

Channels can be unbuffered which means that a writer blocks until a reader is
ready to pick up an item. Unbuffered channels can be useful when the channel is
being used for synchronization, but not when the channel is being used to
increase parallelism.

### Consider lock-free approaches

Sometimes lock-free data structures can make a difference over more conventional
mutex-protected data structures. However, direct atomic variable manipulation
can be dangerous. Prefer higher-level abstractions.

### Use lock-free map to manage a cache of RPC channels.

#### Problem Description
Entries in an RPC stub cache are read thousands of times a second and modified
rarely. Switching to an appropriate lock-free map reduces search latency by
3%-5%.

### Use a fixed lexicon+lock-free hash map to speed-up
determining IsValidTokenId.

#### Problem Description
dynamic_token_class_manager.h

#### Code Diff
```diff
- mutable Mutex mutex_;
- 
- // The density of this hash map is guaranteed by the fact that the
- // dynamic lexicon reuses previously allocated TokenIds before trying
- // to allocate new ones.
- dense_hash_map<TokenId, common::LocalTokenClassId> tid_to_cid_
-     GUARDED_BY(mutex_);
+ // Read accesses to this hash-map should be done using
+ // 'epoch_gc_'::(EnterFast / LeaveFast). The writers should periodically
+ // GC the deleted entries, by simply invoking LockFreeHashMap::CreateGC.
+ typedef util::gtl::LockFreeHashMap<TokenId, common::LocalTokenClassId>
+     TokenIdTokenClassIdMap;
+ TokenIdTokenClassIdMap tid_to_cid_;
```

## Protocol Buffer advice

Protobufs are a convenient representation of data, especially if the data will
be sent over the wire or stored persistently. However, they can have significant
performance costs. For example, a piece of code that fills in a list of 1000
points and then sums up the Y coordinates, speeds up by a factor of 20 when
converted from protobufs to a C++ std::vector of structs!

### Benchmark code for both versions.

#### Problem Description
Protobuf version:
Non-protobuf version:

In addition, the protobuf version adds a few kilobytes of code and data to the
binary, which may not seem like much, but adds up quickly in systems with many
protobuf types. This increased size creates performance problems by creating
i-cache and d-cache pressure.

Here are some tips related to protobuf performance:

### Do not use protobufs unnecessarily.

#### Problem Description
Given the factor of 20 performance difference described above, if some data is
never serialized or parsed, you probably should not put it in a protocol buffer.
The purpose of protocol buffers is to make it easy to serialize and deserialize
data structures, but they can have significant code-size, memory, and CPU
overheads. Do not use them if all you want are some of the other niceties like
DebugString and copyability.

### Avoid unnecessary message hierarchies.

#### Problem Description
Message hierarchy can be useful to organize information in a more readable
fashion. However, the extra level of message hierarchy incurs overheads like
memory allocations, function calls, cache misses, larger serialized messages,
etc.
E.g., instead of:
Prefer:
A protocol buffer message corresponds to a message class in C++ generated code
and emits a tag and the length of the payload on the wire. To carry an integer,
the old form requires more allocations (and deallocations) and emits a larger
amount of generated code. As a result, all protocol buffer operations (parsing,
serialization, size, etc.) become more expensive, having to traverse the message
tree. The new form does not have such overhead and is more efficient.

### Use small field numbers for frequently occurring fields.

#### Problem Description
Protobufs use a variable length integer representation for the combination of
field number and wire format (see the
protobuf encoding documentation).
This representation is 1 byte for field numbers between 1 and 15, and two bytes
for field numbers between 16 and 2047. (Field numbers 2048 or greater should
typically be avoided.)
Consider pre-reserving some small field numbers for future extension of
performance-sensitive protobufs.

### Choose carefully between int32, sint32, fixed32, and uint32 (and
similarly for the 64 bit variants).

#### Problem Description
Generally, use int32 or int64, but use fixed32 or fixed64 for large
values like hash codes and sint32 or sint64 for values are that are often
negative.
A varint occupies fewer bytes to encode small integers and can save space at the
cost of more expensive decoding. However, it can take up more space for negative
or large values. In that case, using fixed32 or fixed64 (instead of uint32 or
uint64) reduces size with much cheaper encoding and decoding. For small negative
integers, use sint32 or sint64 instead of int32 or int64.d

### For proto2, pack repeated numeric fields by annotating them with
[packed=true].

#### Problem Description
In proto2, repeated values are serialized as a sequence of (tag, value) pairs by
default. This is inefficient because tags have to be decoded for every element.
Packed repeated primitives are serialized with the length of the payload first
followed by values without tags. When using fixed-width values, we can avoid
reallocations by knowing the final size the moment we start parsing; i.e., no
reallocation cost. We still don't know how many varints are in the payload and
may have to pay the reallocation cost.
In proto3, repeated fields are packed by default.
Packed works best with fixed-width values like fixed32, fixed64, float, double,
etc. since the entire encoded length can be predetermined by multiplying the
number of elements by the fixed value size, instead of having to calculate the
length of each individual element.

### Use bytes instead for string for binary data
and large values.

#### Problem Description
The string type holds UTF8-encoded text, and can sometimes require validation.
The bytes type can hold an arbitrary sequence of bytes (non-text data) and is
often more appropriate as well as more efficient than string.

### Consider string_type = VIEW to avoid copying.

#### Problem Description
Copying a big string or bytes field during parsing is expensive. Such cost can
often be avoided by marking the field with string_type = VIEW.
Without the VIEW annotation, when the protocol buffer is parsed, the
potentially large field contents are copied from the serialized protocol buffer
to a string object in memory. Depending on the number of string or bytes fields
and the size of those fields, the overhead of copying can be significant.
Instead of copying the big binary blobs, routines like
ParseFromStringWithAliasing use absl::string_view to reference the original
backing string. Note that the backing string (the serialized protocol buffer)
must outlive the protocol buffer instance that contains the alias.

### Consider using Cord for large fields to reduce copying
costs.

#### Problem Description
Annotating large bytes and string fields with [ctype=CORD] may reduce
copying costs. This annotation changes the representation of the field from
std::string to absl::Cord. absl::Cord uses reference counting and
tree-based storage to reduce copying and appending costs. If a protocol buffer
is serialized to a cord, parsing a string or bytes field with [ctype=CORD] can
avoid copying the field contents.
Performance of a Cord field depends on length distribution and access patterns.
Use benchmarks to validate such changes.

### Use protobuf arenas in C++ code.

#### Problem Description
Consider using arenas to save allocation and deallocation costs, especially for
protobufs containing repeated, string, or message fields.
Message and string fields are heap-allocated (even if the top-level protocol
buffer object is stack-allocated). If a protocol buffer message has a lot of sub
message fields and string fields, allocation and deallocation cost can be
significant. Arenas amortize allocation costs and makes deallocation virtually
free. It also improves memory locality by allocating from contiguous chunks of
memory.

### Keep .proto files small

#### Problem Description
Do not put too many messages in a single .proto file. Once you rely on anything
at all from a .proto file, the entire file will get pulled in by the linker even
if it's mostly unused. This increases build times and binary sizes. You can use
extensions and Any to avoid creating hard dependencies on big
.proto files with many message types.

### Consider storing protocol buffers in serialized form, even in memory.

#### Problem Description
In-memory protobuf objects have a large memory footprint (often 5x the wire
format size), potentially spread across many cache lines. So if your application
is going to keep many protobuf objects live for long periods of time, consider
storing them in serialized form.

### Avoid protobuf map fields.

#### Problem Description
Protobuf map fields have performance problems that usually outweigh the small
syntactic convenience they provide. Prefer using non-protobuf maps initialized
from protobuf contents:
msg.proto

### Use protobuf message definition with a subset of the fields.

#### Problem Description
If you want to access only a few fields of a large message type, consider
defining your own protocol buffer message type that mimics the original type,
but only defines the fields that you care about. Here's an example:
By parsing a serialized FullMessage into a SubsetMessage, only two out of a
hundred fields are parsed and others are treated as unknown fields. Consider
using APIs that discard unknown fields to improve performance even more when
appropriate.

### Reuse protobuf objects when possible.

#### Problem Description
Declare protobuf objects outside loops so that their allocated storage can be
reused across loop iterations.

## C++-Specific advice

## C++ 特定建议

### absl::flat_hash_map (and set)

Absl hash tables usually
out-perform C++ standard library containers such as std::map and
std::unordered_map.

Absl 哈希表通常
优于 C++ 标准库容器，如 std::map 和
std::unordered_map。

### Speed up LanguageFromCode (use absl::flat_hash_map
instead of a __gnu_cxx::hash_map).

#### Problem Description
languages.cc
Benchmark results:

#### Code Diff
```diff
- class CodeToLanguage
-     ...
-     : public __gnu_cxx::hash_map<absl::string_view, i18n::languages::Language,
-                                  CodeHash, CodeCompare> {
+ class CodeToLanguage
+     ...
+     : public absl::flat_hash_map<absl::string_view, i18n::languages::Language,
+                                  CodeHash, CodeCompare> {
```

### Speed up stats publish/unpublish (an older change, so
uses dense_hash_map instead of absl::flat_hash_map, which did not exist at the
time).

#### Problem Description
publish.cc

#### Code Diff
```diff
- typedef hash_map<uint64, Publication*> PublicationMap;
- static PublicationMap* publications = NULL;
+ typedef dense_hash_map<uint64, Publication*> PublicationMap;;
+ static PublicationMap* publications GUARDED_BY(mu) = NULL;
```

### Use dense_hash_map instead of hash_map for keeping track of
SelectServer alarms (would use absl::flat_hash_map today).

#### Problem Description
alarmer.h

#### Code Diff
```diff
- typedef hash_map<int, Alarm*> AlarmList;
+ typedef dense_hash_map<int, Alarm*> AlarmList;
```

### absl::btree_map/absl::btree_set

absl::btree_map and absl::btree_set store multiple entries per tree node. This
has a number of advantages over ordered C++ standard library containers such as
std::map. First, the pointer overhead of pointing to child tree nodes is often
significantly reduced. Second, because the entries or key/values are stored
consecutively in memory for a given btree tree node, cache efficiency is often
significantly better.

### Use btree_set instead of std::set to represent a very heavily used
work-queue.

#### Problem Description
register_allocator.h

### util::bitmap::InlinedBitVector

util::bitmap::InlinedBitvector can store short bit-vectors inline, and
therefore can often be a better choice than std::vector<bool> or other bitmap
types.

### Use InlinedBitVector instead of std::vector<bool>, and
then use FindNextBitSet to find the next item of interest.

#### Problem Description
block_encoder.cc

#### Code Diff
```diff
- vector<bool> live_reads(nreads);
- ...
- for (int offset = 0; offset < b_.block_width(); offset++) {
-   ...
-   for (int r = 0; r < nreads; r++) {
-     if (live_reads[r]) {
+ util::bitmap::InlinedBitVector<4096> live_reads(nreads);
+ ...
+ for (int offset = 0; offset < b_.block_width(); offset++) {
+   ...
+   for (size_t r = 0; live_reads.FindNextSetBit(&r); r++) {
+     DCHECK(live_reads[r]);
```

### absl::InlinedVector

absl::InlinedVector stores a small number of elements inline (configurable via
the second template argument). This enables small vectors up to this number of
elements to generally have better cache efficiency and also to avoid allocating
a backing store array at all when the number of elements is small.

### Use InlinedVector instead of std::vector in various places.

#### Problem Description
bundle.h

### gtl::vector32

Saves space by using a customized vector type that only supports sizes that fit
in 32 bits.

### Simple type change saves ~8TiB of memory in Spanner.

#### Problem Description
table_ply.h

#### Code Diff
```diff
- class TablePly {
-     ...
-     // Returns the set of data columns stored in this file for this table.
-     const std::vector<FamilyId>& modified_data_columns() const {
-       return modified_data_columns_;
-     }
-     ...
-    private:
-     ...
-     std::vector<FamilyId> modified_data_columns_;  // Data columns in the table.
+ #include "util/gtl/vector32.h"
+     ...
+     // Returns the set of data columns stored in this file for this table.
+     absl::Span<const FamilyId> modified_data_columns() const {
+       return modified_data_columns_;
+     }
+     ...
+ 
+     ...
+     // Data columns in the table.
+     gtl::vector32<FamilyId> modified_data_columns_;
```

### gtl::small_map

gtl::small_map uses an inline array to store up to a certain number of unique
key-value-pair elements, but upgrades itself automatically to be backed by a
user-specified map type when it runs out of space.

### Use gtl::small_map in tflite_model.

#### Problem Description
tflite_model.cc

#### Code Diff
```diff
- using ChoiceIdToContextMap = gtl::flat_hash_map<int, TFLiteContext*>;
+ using ChoiceIdToContextMap =
+     gtl::small_map<gtl::flat_hash_map<int, TFLiteContext*>>;
```

### gtl::small_ordered_set

gtl::small_ordered_set is an optimization for associative containers (such as
std::set or absl::btree_multiset). It uses a fixed array to store a certain
number of elements, then reverts to using a set or multiset when it runs out of
space. For sets that are typically small, this can be considerably faster than
using something like set directly, as set is optimized for large data sets. This
change shrinks cache footprint and reduces critical section length.

### Use gtl::small_ordered_set to hold set of listeners.

#### Problem Description
broadcast_stream.h

#### Code Diff
```diff
- class BroadcastStream : public ParsedRtpTransport {
-  ...
-  private:
-   ...
-   std::set<ParsedRtpTransport*> listeners_ ABSL_GUARDED_BY(listeners_mutex_);
- };
+ class BroadcastStream : public ParsedRtpTransport {
+  ...
+  private:
+   ...
+   using ListenersSet =
+       gtl::small_ordered_set<std::set<ParsedRtpTransport*>, 10>;
+   ListenersSet listeners_ ABSL_GUARDED_BY(listeners_mutex_);
```

### gtl::intrusive_list

gtl::intrusive_list<T> is a doubly-linked list where the link pointers are
embedded in the elements of type T. It saves one cache line+indirection per
element when compared to std::list<T*>.

### Use intrusive_list to keep track of inflight requests for
each index row update.

#### Problem Description
row-update-sender-inflight-set.h

#### Code Diff
```diff
- std::set<int64> inflight_requests_ GUARDED_BY(mu_);
+ class SeqNum : public gtl::intrusive_link<SeqNum> {
+   ...
+   int64 val_ = -1;
+   ...
+ };
+ ...
+ gtl::intrusive_list<SeqNum> inflight_requests_ GUARDED_BY(mu_);
```

### Limit absl::Status and absl::StatusOr usage

Even though absl::Status and absl::StatusOr types are fairly efficient, they
have a non-zero overhead even in the success path and should therefore be
avoided for hot routines that don’t need to return any meaningful error details
(or perhaps never even fail!):

### Avoid StatusOr<int64> return type for
RoundUpToAlignment() function.

#### Problem Description
best_fit_allocator.cc
best_fit_allocator.h

#### Code Diff
```diff
- absl::StatusOr<int64> BestFitAllocator::RoundUpToAlignment(int64 bytes) const {
-   TPU_RET_CHECK_GE(bytes, 0);
- 
-   const int64 max_aligned = MathUtil::RoundDownTo<int64>(
-       std::numeric_limits<int64>::max(), alignment_in_bytes_);
-   if (bytes > max_aligned) {
-     return util::ResourceExhaustedErrorBuilder(ABSL_LOC)
-            << "Attempted to allocate "
-            << strings::HumanReadableNumBytes::ToString(bytes)
-            << " which after aligning to "
-            << strings::HumanReadableNumBytes::ToString(alignment_in_bytes_)
-            << " cannot be expressed as an int64.";
-   }
- 
-   return MathUtil::RoundUpTo<int64>(bytes, alignment_in_bytes_);
- }
+ // Rounds bytes up to nearest multiple of alignment_.
+ // REQUIRES: bytes >= 0.
+ // REQUIRES: result does not overflow int64.
+ // REQUIRES: alignment_in_bytes_ is a power of 2 (checked in constructor).
+ int64 RoundUpToAlignment(int64 bytes) const {
+   DCHECK_GE(bytes, 0);
+   DCHECK_LE(bytes, max_aligned_bytes_);
+   int64 result =
+       ((bytes + (alignment_in_bytes_ - 1)) & ~(alignment_in_bytes_ - 1));
+   DCHECK_EQ(result, MathUtil::RoundUpTo<int64>(bytes, alignment_in_bytes_));
+   return result;
+ }
```

### Add ShapeUtil::ForEachIndexNoStatus to avoid creating a
Status return object for every element of a tensor.

#### Problem Description
shape_util.h
literal.cc

#### Code Diff
```diff
- using ForEachVisitorFunction =
-     absl::FunctionRef<StatusOr<bool>(absl::Span<const int64_t>)>;
-     ...
- static void ForEachIndex(const Shape& shape, absl::Span<const int64_t> base,
-                          absl::Span<const int64_t> count,
-                          absl::Span<const int64_t> incr,
-                          const ForEachVisitorFunction& visitor_function);
+ using ForEachVisitorFunctionNoStatus =
+     absl::FunctionRef<bool(absl::Span<const int64_t>)>;
+     ...
+ static void ForEachIndexNoStatus(
+     const Shape& shape, absl::Span<const int64_t> base,
+     absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
+     const ForEachVisitorFunctionNoStatus& visitor_function);
```

### In TF_CHECK_OK, avoid creating Ok object in order to test
for ok().

#### Problem Description
status.h

#### Code Diff
```diff
- #define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))
- #define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))
+ extern tensorflow::string* TfCheckOpHelperOutOfLine(
+     const ::tensorflow::Status& v, const char* msg);
+ inline tensorflow::string* TfCheckOpHelper(::tensorflow::Status v,
+                                            const char* msg) {
+   if (v.ok()) return nullptr;
+   return TfCheckOpHelperOutOfLine(v, msg);
+ }
+ #define TF_CHECK_OK(val)                                           \
+   while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \
+   LOG(FATAL) << *(_result)
+ #define TF_QCHECK_OK(val)                                          \
+   while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \
+   LOG(QFATAL) << *(_result)
```

### Remove StatusOr from the hot path of remote procedure
calls (RPCs).

#### Problem Description
Removal of StatusOr from a hot path eliminated a 14% CPU regression in RPC
benchmarks caused by an earlier change.
privacy_context.h
privacy_context_statusfree.h

#### Code Diff
```diff
- absl::StatusOr<privacy::context::PrivacyContext> GetRawPrivacyContext(
-     const CensusHandle& h);
+ enum class Result {
+   kSuccess,
+   kNoRootScopedData,
+   kNoPrivacyContext,
+   kNoDDTContext,
+   kDeclassified,
+   kNoPrequestContext
+ };
+ ...
+ Result GetRawPrivacyContext(const CensusHandle& h,
+                             PrivacyContext* privacy_context);
```

## Bulk operations

If possible, handle many items at once rather than just one at a time.

### absl::flat_hash_map compares one hash byte per key from a
group of keys using a single SIMD instruction.

#### Problem Description
See Swiss Table Design Notes and
related CppCon 2017 and
CppCon 2019 talks by Matt
Kulukundis.
raw_hash_set.h

#### Code Diff
```diff
+ // Returns a bitmask representing the positions of slots that match hash.
+ BitMask<uint32_t> Match(h2_t hash) const {
+   auto ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
+   auto match = _mm_set1_epi8(hash);
+   return BitMask<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(match, ctrl)));
+ }
```

### Do single operations to deal with many bytes and fix
things up, rather than checking every byte what to do.

#### Problem Description
ordered-code.cc

#### Code Diff
```diff
- int len = 0;
- while (val > 0) {
-   len++;
-   buf[9 - len] = (val & 0xff);
-   val >>= 8;
- }
- buf[9 - len - 1] = (unsigned char)len;
- len++;
- FastStringAppend(dest, reinterpret_cast<const char*>(buf + 9 - len), len);
+ BigEndian::Store(val, buf + 1);  // buf[0] may be needed for length
+ const unsigned int length = OrderedNumLength(val);
+ char* start = buf + 9 - length - 1;
+ *start = length;
+ AppendUpto9(dest, start, length + 1);
```

### Improve Reed-Solomon processing speed by handling
multiple interleaved input buffers more efficiently in chunks.

### Decode four integers at a time (circa 2004).

#### Problem Description
Introduced a
GroupVarInt format
that encodes/decodes groups of 4 variable-length integers at a time in 5-17
bytes, rather than one integer at a time. Decoding one group of 4 integers in
the new format takes ~1/3rd the time of decoding 4 individually varint-encoded
integers.
groupvarint.cc

#### Code Diff
```diff
+ const char* DecodeGroupVar(const char* p, int N, uint32* dest) {
+   assert(groupvar_initialized);
+   assert(N % 4 == 0);
+   while (N) {
+     uint8 tag = *p;
+     p++;
+ 
+     uint8* lenptr = &groupvar_table[tag].length[0];
+ 
+ #define GET_NEXT                                        \
+     do {                                                \
+       uint8 len = *lenptr;                              \
+       *dest = UNALIGNED_LOAD32(p) & groupvar_mask[len]; \
+       dest++;                                           \
+       p += len;                                         \
+       lenptr++;                                         \
+     } while (0)
+     GET_NEXT;
+     GET_NEXT;
+     GET_NEXT;
+     GET_NEXT;
+ #undef GET_NEXT
+ 
+     N -= 4;
+   }
+   return p;
+ }
```

### Encode groups of 4 k-bit numbers at a time.

#### Problem Description
Added KBitStreamEncoder and KBitStreamDecoder classes to encode/decode 4 k-bit
numbers at a time into a bit stream. Since K is known at compile time, the
encoding and decoding can be quite efficient. E.g., since four numbers are
encoded at a time, the code can assume that the stream is always byte-aligned
(for even k), or nibble-aligned (for odd k).

## CLs that demonstrate multiple techniques

## 演示多种技术的CL

Sometimes a single CL contains a number of performance-improving changes that
use many of the preceding techniques. Looking at the kinds of changes in these
CLs is sometimes a good way to get in the mindset of making general changes to
speed up the performance of some part of a system after that has been identified
as a bottleneck.

### Speed up GPU memory allocator by ~40%.

#### Problem Description
36-48% speedup in allocation/deallocation speed for GPUBFCAllocator:
Added multi-threaded benchmark to test allocation under contention.
Speeds up ptb_word_lm on my desktop machine with a Titan X card from 8036 words
per second to 8272 words per second (+2.9%).

### Speed up Pathways throughput by ~20% via a set of
miscellaneous changes.

### ~15% XLA compiler performance improvement through a
series of changes.

#### Problem Description
Some changes to speed up XLA compilation:
Overall speedup of 14% in XLA compilation time for one important
model.

### Speed up low level logging in Google Meet application
code.

#### Problem Description
Speed up ScopedLogId, which is on the critical path for each packet.

### Reduce XLA compilation time by ~31% by improving Shape
handling.

#### Problem Description
Several changes to improve XLA compiler performance:
The newly added ForEachIndexNoStatus is considerably faster than the
ForEachIndex variant (it only exists in this new cl, but the benchmark work that
is done by BM_ForEachIndexNoStatus/NUM is comparable to the BM_ForEachIndex/NUM
results above).
Broadcast performance improves by ~58%.
Macro results from doing ahead-of-time compilation of a large language model
(program does more than just the XLA compilation, but spends a bit less than
half its time in XLA-related code):
Baseline program overall: 573 seconds With this cl program overall: 465 seconds
(+19% improvement)
Time spent in compiling the two largest XLA programs in running this program:
Baseline: 141s + 143s = 284s With this CL: 99s + 95s = 194s (+31% improvement)

### Reduce compilation time for large programs by ~22% in
Plaque (a distributed execution framework).

#### Problem Description
Small tweaks to speed up compilation by ~22%.
Measurement of speed on large programs (~45K ops):

### MapReduce improvements (~2X speedup for wordcount
benchmark).

#### Problem Description
Mapreduce speedups:
Reduces time for one wordcount benchmark from 12.56s to 6.55s.

### Rework the alarm handling code in the SelectServer to
significantly improve its performance (adding+removing an alarm from 771 ns to
271 ns).

#### Problem Description
Reworked the alarm handling code in the SelectServer to significantly improve
its performance.
Changes:
Benchmark results
With this change

### 3.3X performance in index serving speed!

#### Problem Description
We found a number of performance issues when planning a switch from on-disk to
in-memory index serving in 2001. This change fixed many of these problems and
took us from 150 to over 500 in-memory queries per second (for a 2 GB in-memory
index on dual processor Pentium III machine).

## Further reading

In no particular order, a list of performance related books and articles that
the authors have found helpful:

<ul>
<li>Optimizing software in C++
by Agner Fog. Describes many useful low-level techniques for improving
performance.</li>
<li>Understanding Software Dynamics
by Richard L. Sites. Covers expert methods and advanced tools for diagnosing
and fixing performance problems.</li>
<li>Performance tips of the week - a collection of
useful tips.</li>
<li>Performance Matters - a collection of
articles about performance.</li>
<li>Daniel Lemire’s blog - high performance
implementations of interesting algorithms.</li>
<li>Building Software Systems at Google and Lessons Learned -
a video that describes system performance issues encountered at Google over
a decade.</li>
<li>Programming Pearls
and
More Programming Pearls: Confessions of a Coder
by Jon Bentley. Essays on starting with algorithms and ending up with simple
and efficient implementations.</li>
<li>Hacker’s Delight by
Henry S. Warren. Bit-level and arithmetic algorithms for solving some common
problems.</li>
<li>Computer Architecture: A Quantitative Approach
by John L. Hennessy and David A. Patterson - Covers many aspects of computer
architecture, including one that performance-minded software developers
should be aware of like caches, branch predictors, TLBs, etc.</li>
</ul>

## Suggested citation

If you want to cite this document, we suggest:

Or in BibTeX:

## Acknowledgments

Many colleagues have provided helpful feedback on this document, including:

<ul>
<li>Adrian Ulrich</li>
<li>Alexander Kuzmin</li>
<li>Alexei Bendebury</li>
<li>Alexey Alexandrov</li>
<li>Amer Diwan</li>
<li>Austin Sims</li>
<li>Benoit Boissinot</li>
<li>Brooks Moses</li>
<li>Chris Kennelly</li>
<li>Chris Ruemmler</li>
<li>Danila Kutenin</li>
<li>Darryl Gove</li>
<li>David Majnemer</li>
<li>Dmitry Vyukov</li>
<li>Emanuel Taropa</li>
<li>Felix Broberg</li>
<li>Francis Birck Moreira</li>
<li>Gideon Glass</li>
<li>Henrik Stewenius</li>
<li>Jeremy Dorfman</li>
<li>John Dethridge</li>
<li>Kurt Kluever</li>
<li>Kyle Konrad</li>
<li>Lucas Pereira</li>
<li>Marc Eaddy</li>
<li>Michael Marty</li>
<li>Michael Whittaker</li>
<li>Mircea Trofin</li>
<li>Misha Brukman</li>
<li>Nicolas Hillegeer</li>
<li>Ranjit Mathew</li>
<li>Rasmus Larsen</li>
<li>Soheil Hassas Yeganeh</li>
<li>Srdjan Petrovic</li>
<li>Steinar H. Gunderson</li>
<li>Stergios Stergiou</li>
<li>Steven Timotius</li>
<li>Sylvain Vignaud</li>
<li>Thomas Etter</li>
<li>Thomas Köppe</li>
<li>Tim Chestnutt</li>
<li>Todd Lipcon</li>
<li>Vance Lankhaar</li>
<li>Victor Costan</li>
<li>Yao Zuo</li>
<li>Zhou Fang</li>
<li>Zuguang Yang</li>
</ul>

