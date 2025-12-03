# URGENT: Embedding Generation is Broken!

**Date:** 2025-11-21
**Severity:** CRITICAL - Search returns wrong results
**Status:** ❌ BLOCKING PRODUCTION

---

## The Problem

Server logs show:

```
ERROR ort::logging: Non-zero status code returned while running Gather node.
Name:'/embeddings/token_type_embeddings/Gather'
Status Message: Missing Input: token_type_ids

WARN shodh_memory::embeddings::minilm: ONNX inference failed... Falling back to simplified.
```

**What this means:**
- ONNX model expects `token_type_ids` input
- Tokenizer is NOT providing it
- Model falls back to "simplified" mode
- Embeddings are either WRONG or RANDOM
- This explains why search returns incorrect results!

---

## Impact on Tests

### Before Fix: Search Worked Partially (4/14 tests passed)
### After Indexing Fix: Search STILL Broken (11/14 tests passed, but search returns wrong results)

**Example Failure:**
```
Query: "obstacle"
Expected: "The robot detected an obstacle..."
Got: "Battery level dropped to 15 percent" ❌ WRONG!
```

**Why indexing fix didn't help:**
- Indexing is working now (immediate indexing ✅)
- But embeddings being indexed are BROKEN
- Garbage in → Garbage out

---

## Root Cause

Location: `src/embeddings/minilm.rs` (likely)

**The tokenizer output is missing `token_type_ids`**, which BERT-based models (like MiniLM) need.

### What token_type_ids Are

In BERT/MiniLM:
- `input_ids`: Token indices
- `attention_mask`: Which tokens to attend to
- `token_type_ids`: Segment IDs (0 for sentence A, 1 for sentence B)

For single-sentence encoding, `token_type_ids` should be all zeros.

---

## The Fix

### Option 1: Add token_type_ids to Tokenizer Output

```rust
// In src/embeddings/minilm.rs or wherever tokenizer is called

let tokens = self.tokenizer.encode(&text)?;

// ADD THIS:
let token_type_ids = vec![0i64; tokens.len()]; // All zeros for single sentence

// Pass to ONNX model:
let inputs = vec![
    ("input_ids", input_ids_tensor),
    ("attention_mask", attention_mask_tensor),
    ("token_type_ids", token_type_ids_tensor), // ← ADD THIS
];
```

### Option 2: Use a Simplified ONNX Model

Export MiniLM without requiring token_type_ids:

```python
# Re-export model with default token_type_ids
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Export with inputs that include default token_type_ids
# Or export a version that doesn't require it
```

### Option 3: Check "Simplified" Fallback

The code mentions "Falling back to simplified" - check what this does:

```rust
// Find this in src/embeddings/minilm.rs
// If "simplified" mode exists, it might be using random/dummy embeddings
```

---

## Files to Check

1. **`src/embeddings/minilm.rs`**
   - Check `encode()` method
   - Look for tokenizer usage
   - Find ONNX model input preparation

2. **`src/embeddings/mod.rs`**
   - Check Embedder trait implementation

3. **ONNX Model File**
   - Check if model actually requires token_type_ids
   - Might need to re-export model

---

## Testing the Fix

After fixing, verify:

1. **No more ONNX errors in logs:**
   ```
   ✅ Should NOT see: "Missing Input: token_type_ids"
   ✅ Should NOT see: "Falling back to simplified"
   ```

2. **Embedding generation works:**
   ```bash
   # Add debug logging to print first few embedding values
   # Should be different for different texts
   ```

3. **Search returns correct results:**
   ```bash
   cd shodh-memory-python
   python diagnose_search.py

   # Should show:
   # Query: "robot obstacle"
   # Top result: "The robot detected an obstacle..." ✅
   ```

4. **Benchmarks pass:**
   ```bash
   cd benchmarks
   python accuracy_benchmark.py

   # Should show:
   # Retrieval Accuracy: >80%
   # Answer Coverage: >80%
   ```

---

## Current Workaround

I've fixed the Python benchmark to handle None scores (temporary fix):

```python
# benchmarks/accuracy_benchmark.py line 115
score = top_result.score if top_result.score is not None else 0.0
```

This prevents crashes but doesn't fix the underlying embedding problem.

---

## Priority

**CRITICAL - FIX IMMEDIATELY**

All other fixes (delete endpoint, score calculation, etc.) are meaningless if embeddings don't work.

**Search is the core functionality!**

---

## Next Steps

1. ✅ Score field crash fixed (Python workaround)
2. ❌ **FIX EMBEDDING GENERATION** (Rust backend) ← DO THIS NOW
3. ⏳ Fix delete endpoint (after embeddings work)
4. ⏳ Run benchmarks to verify

---

## Debugging Commands

```bash
# Check ONNX model inputs
cd models/minilm-l6
python -c "
import onnx
model = onnx.load('model_quantized.onnx')
print('Required inputs:')
for inp in model.graph.input:
    print(f'  - {inp.name}: {inp.type}')
"

# Test embeddings directly
cd shodh-memory
cargo test minilm -- --nocapture

# Watch logs while testing
tail -f logs/shodh-memory.log | grep -i "onnx\|embedding"
```

---

## References

- MiniLM Paper: https://arxiv.org/abs/2002.10957
- BERT Input Format: https://huggingface.co/docs/transformers/model_doc/bert#inputs
- ONNX Runtime Rust: https://docs.rs/ort/latest/ort/
