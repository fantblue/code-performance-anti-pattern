# Performance Tuning Tips from Jeff Dean's Article

## The importance of thinking about performance

## Estimation

## Measurement

## API considerations

### Add bulk MemoryManager::LookupMany interface.

#### Problem Description
In addition to adding a bulk interface, this also simplified the signature for
the new bulk variant: it turns out clients only needed to know if all the keys
were found, so we can return a bool rather than a Status object.
memory_manager.h

#### Improvement
See code diff below.

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

#### Problem Description
object_store.h
memory_tracking.cc

#### Improvement
See code diff below.

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

#### Problem Description
Bulk initialization of a heap can be done in O(N) time, whereas adding one
element at a time and updating the heap property after each addition requires
O(N lg(N)) time.

### Cache block decode results for use in future calls.

#### Problem Description
Each lookup needs to decode a whole block of K entries. Store the decoded
entries in a cache and consult the cache on future lookups.
lexicon.cc

#### Improvement
See code diff below.

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

### Add RPC_Stats::RecordRPC variant allowing client to pass in
already available WallTime value.

#### Problem Description
rpc-stats.h
clientchannel.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m);
+ static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m,
+                       WallTime now);
```

### Make a class thread-compatible since callers are already
synchronized.

#### Problem Description
hitless-transfer-phase.cc
hitless-transfer-phase.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- TransferPhase HitlessTransferPhase::get() const {
-   static CallsiteMetrics cm("HitlessTransferPhase::get");
-   MonitoredMutexLock l(&cm, &mutex_);
-   return phase_;
- }
+ TransferPhase HitlessTransferPhase::get() const { return phase_; }
```

## Algorithmic improvements

### Add nodes to cycle detection structure in reverse
post-order.

#### Problem Description
We were previously adding graph nodes and edges one at a time to a
cycle-detection data structure, which required expensive work per edge. We now
add the entire graph in reverse post-order, which makes cycle-detection trivial.
graphcycles.h
graphcycles.cc
graph_partitioner.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Reduce allocations and improve cache footprint by
converting btree<a,btree<b,c>> to btree<pair<a,b>,c>.

#### Problem Description
graph_splitter.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- absl::btree_map<std::string, absl::btree_map<std::string, OpDef>> ops;
+ // The btree maps from {package_name, op_name} to its const Opdef*.
+ absl::btree_map<std::pair<absl::string_view, absl::string_view>,
+                 const OpDef*>
+     ops;
```

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

### Use an array instead of flat_map.

#### Problem Description
rtp_controller.h

#### Improvement
See code diff below.

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

### Spanner placement system. Replace
dense_hash_set<ZoneId> with a bit-vector with one bit per zone.

#### Problem Description
zone_set.h
Benchmark results:

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Reducing allocations increases benchmark throughput by
21%.

#### Problem Description
memory_manager.cc

#### Improvement
See code diff below.

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

### Pre-size a vector and fill it in, rather than N push_back
operations.

#### Problem Description
indexblockdecoder.cc

#### Improvement
See code diff below.

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

### Avoid an extra copy when receiving a tensor via gRPC.

#### Problem Description
A benchmark that sends around 400KB tensors speeds up by ~10-15%:

### Move large options structure rather than copying it.

#### Problem Description
index.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- return search_iterators::DocPLIteratorFactory::Create(opts);
+ return search_iterators::DocPLIteratorFactory::Create(std::move(opts));
```

### Use std::sort instead of std::stable_sort, which avoids
an internal copy inside the stable sort implementation.

#### Problem Description
encoded-vector-hits.h

#### Improvement
See code diff below.

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

### Hoist variable definition outside of loop iteration.

#### Problem Description
autofdo_profile_utils.h

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

## Avoid unnecessary work

### Make fast path cover more common cases.

#### Problem Description
Add handling of trailing single ASCII bytes, rather than only handling multiples
of four bytes with this routine. This avoids calling the slower generic routine
for all-ASCII strings that are, for example, 5 bytes.
utf8statetable.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Precompute a TensorFlow graph execution node property
that allows us to quickly rule out certain unusual cases.

#### Problem Description
executor.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Move bounds computation outside loop.

#### Problem Description
literal_linearizer.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- for (int64 i = 0; i < src_shape.dimensions(dimension_numbers.front());
-      ++i) {
+ int64 dim_front = src_shape.dimensions(dimension_numbers.front());
+ const uint8* src_buffer_data = src_buffer.data();
+ uint8* dst_buffer_data = dst_buffer.data();
+ for (int64 i = 0; i < dim_front; ++i) {
```

### Defer GetSubSharding call until needed, which reduces 43
seconds of CPU time to 2 seconds.

#### Problem Description
sharding_propagation.cc

#### Improvement
See code diff below.

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

#### Problem Description
Do not update stats on the very frequent allocation/deallocation calls. Instead,
compute stats on demand when the much less frequently called Stats() method is
invoked.

### Preallocate 10 nodes not 200 for query handling in Google's
web server.

#### Problem Description
A simple change that reduced web server's CPU usage by 7.5%.
querytree.h

#### Improvement
See code diff below.

#### Code Diff
```diff
- static const int kInitParseTreeSize = 200;   // initial size of querynode pool
+ static const int kInitParseTreeSize = 10;   // initial size of querynode pool
```

### Change search order for 19% throughput improvement.

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

### Custom printing code for Histogram class is 4x as fast as
sprintf.

#### Problem Description
This code is performance sensitive because it is invoked when monitoring systems
gather statistics from various servers.
histogram_export.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Cache based on precomputed fingerprint of large
serialized proto.

#### Problem Description
dp_ops.cc

#### Improvement
See code diff below.

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

### Speed up ShapeUtil::ForEachState by replacing absl::Span
with raw pointers to the underlying arrays.

#### Problem Description
shape_util.h

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Stop maintaining expensive stats about number of alarms and
closures in SelectServer.

#### Problem Description
Part of changes that reduce time for setting an alarm from 771 ns to 271 ns.
selectserver.h
/selectserver.cc
/selectserver.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Remove logging from guts of memory allocator.

#### Problem Description
This was a small part of a larger change.
gpu_bfc_allocator.cc

### Precompute whether or not logging is enabled outside a
nested loop.

#### Problem Description
image_similarity.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Speed up TF_CHECK_OK.

#### Problem Description
Avoid creating Ok object, and save code space by doing complex formatting of
fatal error message out of line instead of at every call site.
status.h
status.cc

#### Improvement
See code diff below.

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

#### Problem Description


### Improve performance of CHECK_GE by 4.5X and shrink code
size from 125 bytes to 77 bytes.

#### Problem Description
logging.h
logging.cc

#### Improvement
See code diff below.

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

### Reduce absl::flat_hash_set and absl::flat_hash_map code
size.

#### Problem Description
Reduces sizes of some large binaries by ~0.5%.

### Do not inline string allocation and deallocation when not
using protobuf arenas.

#### Problem Description
public/arenastring.h
internal/arenastring.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Replace template argument with a regular argument.

#### Problem Description
Changed a large routine templated on a bool to instead take the bool as an extra
argument. (The bool was only being used once to select one of two string
constants, so a run-time check was just fine.) This reduced the # of
instantiations of the large routine from 287 to 143.
sharding_util_ops.cc

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Problem Description


### Turn many map insertion calls in a row to initialize a
hash table of emoji characters into a single bulk insert operation (188KB of
text down to 360 bytes in library linked into many binaries). 😊

#### Problem Description
textfallback_init.h

#### Improvement
See code diff below.

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

### Four-way parallelization improves the rate of encoding
tokens by ~3.6x.

#### Problem Description
blocked-token-coder.cc

### Parallelization improves decoding performance by 5x.

#### Problem Description
coding.cc

### Acquire lock once to free entire tree of query nodes, rather
than reacquiring lock for every node in tree.

#### Problem Description
mustang-query.cc

#### Improvement
See code diff below.

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

### Reduce number of cache lines touched in critical section.

#### Problem Description
Careful data structure adjustments reduce the number of cache lines accessed
significantly and improve the performance of an ML training run by 3.3%.

### Avoid RPC while holding Mutex.

#### Problem Description
trainer.cc

#### Improvement
See code diff below.

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

### Shard a cache 16 ways which improves throughput under a
multi-threaded load by ~2x.

#### Problem Description
cache.cc

### Shard spanner data structure for tracking calls.

#### Problem Description
transaction_manager.cc

#### Improvement
See code diff below.

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

### Fix information used for shard selection to prevent hash
table issues.

#### Problem Description
netmon_map_impl.h

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### Segregate commonly mutated fields in a different cache
line than other fields.

#### Problem Description
histogram.h

#### Improvement
See code diff below.

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

### Process small work items inline instead of on device
thread pool.

#### Problem Description
cast_op.cc

### Use lock-free map to manage a cache of RPC channels.

#### Problem Description
Entries in an RPC stub cache are read thousands of times a second and modified
rarely. Switching to an appropriate lock-free map reduces search latency by
3%-5%.

### Use a fixed lexicon+lock-free hash map to speed-up
determining IsValidTokenId.

#### Problem Description
dynamic_token_class_manager.h

#### Improvement
See code diff below.

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

### Benchmark code for both versions.

#### Problem Description
Protobuf version:
Non-protobuf version:

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

### Speed up LanguageFromCode (use absl::flat_hash_map
instead of a __gnu_cxx::hash_map).

#### Problem Description
languages.cc
Benchmark results:

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

#### Code Diff
```diff
- typedef hash_map<int, Alarm*> AlarmList;
+ typedef dense_hash_map<int, Alarm*> AlarmList;
```

### Use btree_set instead of std::set to represent a very heavily used
work-queue.

#### Problem Description
register_allocator.h

### Use InlinedBitVector instead of std::vector<bool>, and
then use FindNextBitSet to find the next item of interest.

#### Problem Description
block_encoder.cc

#### Improvement
See code diff below.

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

### Use InlinedVector instead of std::vector in various places.

#### Problem Description
bundle.h

### Simple type change saves ~8TiB of memory in Spanner.

#### Problem Description
table_ply.h

#### Improvement
See code diff below.

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

### Use gtl::small_map in tflite_model.

#### Problem Description
tflite_model.cc

#### Improvement
See code diff below.

#### Code Diff
```diff
- using ChoiceIdToContextMap = gtl::flat_hash_map<int, TFLiteContext*>;
+ using ChoiceIdToContextMap =
+     gtl::small_map<gtl::flat_hash_map<int, TFLiteContext*>>;
```

### Use gtl::small_ordered_set to hold set of listeners.

#### Problem Description
broadcast_stream.h

#### Improvement
See code diff below.

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

### Use intrusive_list to keep track of inflight requests for
each index row update.

#### Problem Description
row-update-sender-inflight-set.h

#### Improvement
See code diff below.

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

### Avoid StatusOr<int64> return type for
RoundUpToAlignment() function.

#### Problem Description
best_fit_allocator.cc
best_fit_allocator.h

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

#### Improvement
See code diff below.

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

### absl::flat_hash_map compares one hash byte per key from a
group of keys using a single SIMD instruction.

#### Problem Description
See Swiss Table Design Notes and
related CppCon 2017 and
CppCon 2019 talks by Matt
Kulukundis.
raw_hash_set.h

### Do single operations to deal with many bytes and fix
things up, rather than checking every byte what to do.

#### Problem Description
ordered-code.cc

#### Improvement
See code diff below.

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

#### Problem Description


### Decode four integers at a time (circa 2004).

#### Problem Description
Introduced a
GroupVarInt format
that encodes/decodes groups of 4 variable-length integers at a time in 5-17
bytes, rather than one integer at a time. Decoding one group of 4 integers in
the new format takes ~1/3rd the time of decoding 4 individually varint-encoded
integers.
groupvarint.cc

### Encode groups of 4 k-bit numbers at a time.

#### Problem Description
Added KBitStreamEncoder and KBitStreamDecoder classes to encode/decode 4 k-bit
numbers at a time into a bit stream. Since K is known at compile time, the
encoding and decoding can be quite efficient. E.g., since four numbers are
encoded at a time, the code can assume that the stream is always byte-aligned
(for even k), or nibble-aligned (for odd k).

## CLs that demonstrate multiple techniques

### Speed up GPU memory allocator by ~40%.

#### Problem Description
36-48% speedup in allocation/deallocation speed for GPUBFCAllocator:
Added multi-threaded benchmark to test allocation under contention.
Speeds up ptb_word_lm on my desktop machine with a Titan X card from 8036 words
per second to 8272 words per second (+2.9%).

### Speed up Pathways throughput by ~20% via a set of
miscellaneous changes.

#### Problem Description


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

