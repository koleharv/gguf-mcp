// Builders for synthetic-but-valid model files, so tests need no downloads.

function u32(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n);
  return b;
}

function u64(n: number): Buffer {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
}

function ggufString(s: string): Buffer {
  const bytes = Buffer.from(s, "utf8");
  return Buffer.concat([u64(bytes.length), bytes]);
}

const T = { UINT32: 4, STRING: 8, ARRAY: 9 } as const;

function kvU32(key: string, value: number): Buffer {
  return Buffer.concat([ggufString(key), u32(T.UINT32), u32(value)]);
}

function kvString(key: string, value: string): Buffer {
  return Buffer.concat([ggufString(key), u32(T.STRING), ggufString(value)]);
}

function kvStringArray(key: string, values: string[]): Buffer {
  return Buffer.concat([
    ggufString(key),
    u32(T.ARRAY),
    u32(T.STRING),
    u64(values.length),
    ...values.map(ggufString),
  ]);
}

function tensorInfo(name: string, shape: number[], type: number): Buffer {
  return Buffer.concat([ggufString(name), u32(shape.length), ...shape.map(u64), u32(type), u64(0)]);
}

export const TINY_GGUF_WEIGHTS = 128;

/** 2 tensors (64x64 F32 + 64x100 F16 = 10,496 params), llama metadata, 7-token vocab. */
export function buildTinyGguf(): Buffer {
  const kvs = [
    kvString("general.architecture", "llama"),
    kvString("general.name", "TestLlama"),
    kvU32("general.file_type", 15), // Q4_K_M
    kvU32("general.alignment", 32),
    kvU32("llama.context_length", 4096),
    kvU32("llama.block_count", 2),
    kvU32("llama.attention.head_count", 4),
    kvU32("llama.attention.head_count_kv", 2),
    kvU32("llama.embedding_length", 64),
    kvStringArray("tokenizer.ggml.tokens", ["a", "b", "c", "d", "e", "f", "g"]),
  ];
  const tensors = [
    tensorInfo("blk.0.attn_q.weight", [64, 64], 0), // F32
    tensorInfo("output.weight", [64, 100], 1), // F16
  ];
  const header = Buffer.concat([
    Buffer.from("GGUF", "latin1"),
    u32(3), // version
    u64(tensors.length),
    u64(kvs.length),
    ...kvs,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (header.length % 32)) % 32);
  return Buffer.concat([header, padding, Buffer.alloc(TINY_GGUF_WEIGHTS, 7)]);
}

/** Two tensors, 205 params, 420 data bytes, with a __metadata__ block. */
export function buildTinySafetensors(): Buffer {
  const header = JSON.stringify({
    __metadata__: { format: "pt" },
    emb: { dtype: "F16", shape: [10, 20], data_offsets: [0, 400] },
    w: { dtype: "F32", shape: [5], data_offsets: [400, 420] },
  });
  const headerBytes = Buffer.from(header, "utf8");
  return Buffer.concat([u64(headerBytes.length), headerBytes, Buffer.alloc(420, 1)]);
}

/** One-tensor shard for index tests: `name`, F32, given shape. */
export function buildShard(name: string, shape: number[]): Buffer {
  const bytes = 4 * shape.reduce((a, b) => a * b, 1);
  const header = JSON.stringify({ [name]: { dtype: "F32", shape, data_offsets: [0, bytes] } });
  const headerBytes = Buffer.from(header, "utf8");
  return Buffer.concat([u64(headerBytes.length), headerBytes, Buffer.alloc(bytes, 2)]);
}
