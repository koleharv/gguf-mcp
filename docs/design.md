# gguf-mcp — Design

**Date:** 2026-08-14
**Goal:** Third public portfolio MCP server. Per user direction: not Canada-themed,
and a different engineering genre than the two open-data API wrappers.

## Idea selection

Eight candidates eliminated by niche-checks before this one (all already have
public MCP servers): date/recurrence math (mcp-time-server-node, rrule.net),
OpenAlex scholarly data (6+ servers), deps.dev/OSV dependency intel (Hound),
SPF/DKIM/DMARC audit (several commercial), GTFS transit (gtfs-mcp +
agency-specific), openFDA (many), financial calculators (Digital Calculator,
13 tools), C2PA content credentials (c2pa-mcp).

**Chosen: local model-file inspection (GGUF + safetensors).** Browser viewers
and CLI explorers exist — proving demand — but no MCP server does it. The
audience (people running local models) overlaps heavily with MCP users.

## What it does

Answers, from files already on disk and without loading a single tensor:

- "What is this .gguf? Architecture, quantization, parameter count, context?"
- "Will this model fit in my 12 GB GPU at 8k context?"
- "What tensors are inside, with what shapes and types?"
- "Show me the metadata / chat template."

Only file **headers** are read, so a 70 GB model inspects in milliseconds.

## Architecture

Zero runtime dependencies beyond the MCP SDK and zod. No network at all.

- `src/gguf.ts` — GGUF binary parser: a buffered cursor over a file handle
  that reads the magic/version, typed metadata KVs (13 value types incl.
  nested arrays), and tensor infos. Pure logic, offline-testable.
- `src/safetensors.ts` — safetensors parser (8-byte LE header length + JSON),
  including sharded models via `*.safetensors.index.json`.
- `src/index.ts` — McpServer wiring, four tools, shared error guard.
- `test/*.test.ts` — vitest; synthetic-but-valid files are **built in the
  tests** with a tiny writer helper, so no fixtures and no downloads.
- `scripts/smoke.mjs` — generates tiny model files in a temp dir and drives
  the built server over stdio. Fully self-contained.

## Key design points

- **Header-only reads.** The cursor pulls bytes from the file on demand in
  1 MiB chunks; tensor data is never touched. A sanity cap (256 MiB) guards
  against malformed headers.
- **Array metadata is summarized, never dumped.** A tokenizer vocabulary is
  100k+ strings; the parser keeps element type, count, and the first few
  values. Long strings (chat templates) are truncated with a marker. An LLM
  context is a scarce resource; the server is designed around that.
- **Quantization decoding:** ggml tensor-type enum and llama.cpp `file_type`
  enum mapped to human names (Q4_K_M, IQ2_XS, BF16...), with graceful
  fallback to the raw code for unknown values.
- **VRAM estimate = exact weights + modeled KV cache.** Weights bytes are
  computed exactly from the file layout (file size minus aligned tensor-data
  start). KV cache uses the standard formula (2 × layers × ctx × kv_heads ×
  head_dim × 2 bytes fp16) from GGUF metadata, with context defaulting to
  min(model context, 8192). Reported as parts + total + margin note, never as
  false precision.
- **Sharded safetensors:** an index.json fans out to per-shard header reads;
  a directory path auto-detects index/single-file layouts.

## Tools

| Tool | Purpose |
|------|---------|
| `inspect_model(path)` | One-call summary: format, architecture, params, quant, context, size, tensor count, notable metadata |
| `list_tensors(path, filter?, limit?)` | Tensor names/shapes/types, substring-filterable, capped |
| `estimate_vram(path, context_length?)` | Fit check: weights + KV cache + margin (GGUF); weights-only for safetensors |
| `get_metadata(path, filter?)` | GGUF KV store / safetensors `__metadata__`, summarized safely |

## Error handling

Wrong magic, truncated files, oversized headers, and unsupported versions
produce specific messages (e.g. "not a GGUF file (magic mismatch)") returned
as MCP tool errors. Paths that don't exist say so plainly.

## Out of scope (v1)

Tensor statistics (means/histograms — requires reading data), PyTorch .bin
(pickle; unsafe by design), ONNX, remote HuggingFace lookups (HF has an
official MCP server), model editing of any kind.
