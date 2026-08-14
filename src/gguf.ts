// GGUF header parser (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
// Reads ONLY the header — metadata KVs and tensor infos — never tensor data,
// so inspecting a 70 GB model costs milliseconds and a few MiB of I/O.

import { open, stat } from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";

const GGUF_MAGIC = 0x46554747; // "GGUF" little-endian
const CHUNK = 1 << 20; // 1 MiB reads
const HEADER_CAP = 256 << 20; // sanity cap: no real header exceeds 256 MiB
const ARRAY_SAMPLE = 5; // metadata arrays keep a sample, never the whole vocab

// Incremental little-endian reader over a file handle. skip() is free (pure
// position move); bytes are only pulled from disk when actually decoded.
class Cursor {
  private buf = Buffer.alloc(0);
  private bufStart = 0;
  pos = 0;

  constructor(
    private fh: FileHandle,
    readonly fileSize: number,
  ) {}

  private async view(n: number): Promise<Buffer> {
    if (this.pos + n > this.fileSize) throw new Error("Unexpected end of file while reading GGUF header");
    if (this.pos > HEADER_CAP) throw new Error("GGUF header exceeds sanity cap — file is likely corrupt");
    if (this.pos < this.bufStart || this.pos + n > this.bufStart + this.buf.length) {
      const size = Math.min(Math.max(n, CHUNK), this.fileSize - this.pos);
      const fresh = Buffer.alloc(size);
      const { bytesRead } = await this.fh.read(fresh, 0, size, this.pos);
      if (bytesRead < n) throw new Error("Unexpected end of file while reading GGUF header");
      this.buf = fresh.subarray(0, bytesRead);
      this.bufStart = this.pos;
    }
    const off = this.pos - this.bufStart;
    return this.buf.subarray(off, off + n);
  }

  skip(n: number): void {
    this.pos += n;
  }

  async u8(): Promise<number> {
    const v = (await this.view(1)).readUInt8(0);
    this.pos += 1;
    return v;
  }
  async i8(): Promise<number> {
    const v = (await this.view(1)).readInt8(0);
    this.pos += 1;
    return v;
  }
  async u16(): Promise<number> {
    const v = (await this.view(2)).readUInt16LE(0);
    this.pos += 2;
    return v;
  }
  async i16(): Promise<number> {
    const v = (await this.view(2)).readInt16LE(0);
    this.pos += 2;
    return v;
  }
  async u32(): Promise<number> {
    const v = (await this.view(4)).readUInt32LE(0);
    this.pos += 4;
    return v;
  }
  async i32(): Promise<number> {
    const v = (await this.view(4)).readInt32LE(0);
    this.pos += 4;
    return v;
  }
  async f32(): Promise<number> {
    const v = (await this.view(4)).readFloatLE(0);
    this.pos += 4;
    return v;
  }
  async f64(): Promise<number> {
    const v = (await this.view(8)).readDoubleLE(0);
    this.pos += 8;
    return v;
  }
  async u64(): Promise<number> {
    const v = (await this.view(8)).readBigUInt64LE(0);
    this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`64-bit value too large to represent: ${v}`);
    return Number(v);
  }
  async i64(): Promise<number> {
    const v = (await this.view(8)).readBigInt64LE(0);
    this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new Error(`64-bit value too large to represent: ${v}`);
    }
    return Number(v);
  }
  async string(): Promise<string> {
    const len = await this.u64();
    const v = (await this.view(len)).toString("utf8");
    this.pos += len;
    return v;
  }
}

// gguf_metadata_value_type
enum ValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

const FIXED_SIZE: Partial<Record<ValueType, number>> = {
  [ValueType.UINT8]: 1,
  [ValueType.INT8]: 1,
  [ValueType.BOOL]: 1,
  [ValueType.UINT16]: 2,
  [ValueType.INT16]: 2,
  [ValueType.UINT32]: 4,
  [ValueType.INT32]: 4,
  [ValueType.FLOAT32]: 4,
  [ValueType.UINT64]: 8,
  [ValueType.INT64]: 8,
  [ValueType.FLOAT64]: 8,
};

export interface ArraySummary {
  array: true;
  elementType: string;
  count: number;
  sample: unknown[];
}

export type MetadataValue = string | number | boolean | ArraySummary;

async function readScalar(c: Cursor, type: ValueType): Promise<string | number | boolean> {
  switch (type) {
    case ValueType.UINT8:
      return c.u8();
    case ValueType.INT8:
      return c.i8();
    case ValueType.UINT16:
      return c.u16();
    case ValueType.INT16:
      return c.i16();
    case ValueType.UINT32:
      return c.u32();
    case ValueType.INT32:
      return c.i32();
    case ValueType.FLOAT32:
      return c.f32();
    case ValueType.BOOL:
      return (await c.u8()) !== 0;
    case ValueType.STRING:
      return c.string();
    case ValueType.UINT64:
      return c.u64();
    case ValueType.INT64:
      return c.i64();
    case ValueType.FLOAT64:
      return c.f64();
    default:
      throw new Error(`Unknown GGUF metadata value type: ${type}`);
  }
}

async function readValue(c: Cursor, type: ValueType): Promise<MetadataValue> {
  if (type !== ValueType.ARRAY) return readScalar(c, type);

  const elemType = (await c.u32()) as ValueType;
  const count = await c.u64();
  const keep = Math.min(count, ARRAY_SAMPLE);
  const sample: unknown[] = [];

  const fixed = FIXED_SIZE[elemType];
  if (fixed !== undefined) {
    for (let i = 0; i < keep; i++) sample.push(await readScalar(c, elemType));
    c.skip((count - keep) * fixed); // O(1) — the rest is never read from disk
  } else {
    // Strings (or nested arrays) must be walked to find where they end,
    // but skipped bytes are never pulled from disk.
    for (let i = 0; i < count; i++) {
      if (i < keep) {
        sample.push(await readValue(c, elemType));
      } else if (elemType === ValueType.STRING) {
        c.skip(await c.u64());
      } else {
        await readValue(c, elemType);
      }
    }
  }
  return { array: true, elementType: ValueType[elemType] ?? String(elemType), count, sample };
}

// ggml_type — tensor storage/quantization formats
const GGML_TYPES: Record<number, string> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
  16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
  22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
  29: "IQ1_M", 30: "BF16",
};

// llama.cpp file_type (general.file_type) — the model's overall quant label
const FILE_TYPES: Record<number, string> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
  10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
  16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
  22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
  28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
};

export interface TensorInfo {
  name: string;
  shape: number[];
  type: string;
}

export interface GgufModel {
  format: "gguf";
  version: number;
  fileSizeBytes: number;
  weightsBytes: number;
  paramCount: number;
  metadata: Record<string, MetadataValue>;
  tensors: TensorInfo[];
}

export async function parseGguf(path: string): Promise<GgufModel> {
  const fileSize = (await stat(path)).size;
  const fh = await open(path, "r");
  try {
    const c = new Cursor(fh, fileSize);

    if ((await c.u32()) !== GGUF_MAGIC) throw new Error("Not a GGUF file (magic mismatch)");
    const version = await c.u32();
    if (version > 0xffff) throw new Error("Big-endian GGUF files are not supported");
    if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version: ${version}`);

    const tensorCount = await c.u64();
    const kvCount = await c.u64();

    const metadata: Record<string, MetadataValue> = {};
    for (let i = 0; i < kvCount; i++) {
      const key = await c.string();
      const type = (await c.u32()) as ValueType;
      metadata[key] = await readValue(c, type);
    }

    const tensors: TensorInfo[] = [];
    let paramCount = 0;
    for (let i = 0; i < tensorCount; i++) {
      const name = await c.string();
      const nDims = await c.u32();
      if (nDims > 8) throw new Error(`Tensor "${name}" claims ${nDims} dimensions — file is likely corrupt`);
      const shape: number[] = [];
      for (let d = 0; d < nDims; d++) shape.push(await c.u64());
      const type = await c.u32();
      await c.u64(); // offset within tensor data — not needed for inspection
      tensors.push({ name, shape, type: GGML_TYPES[type] ?? `type_${type}` });
      paramCount += shape.reduce((a, b) => a * b, 1);
    }

    // Tensor data begins at the next alignment boundary after the header;
    // everything from there to EOF is weights, exactly as they'd sit in memory.
    const alignment = typeof metadata["general.alignment"] === "number" ? (metadata["general.alignment"] as number) : 32;
    const tensorDataStart = Math.ceil(c.pos / alignment) * alignment;

    return {
      format: "gguf",
      version,
      fileSizeBytes: fileSize,
      weightsBytes: fileSize - tensorDataStart,
      paramCount,
      metadata,
      tensors,
    };
  } finally {
    await fh.close();
  }
}

// ---------------------------------------------------------------------------
// Human-facing summaries

export function humanCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function humanBytes(n: number): string {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GiB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MiB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`;
  return `${n} B`;
}

function num(v: MetadataValue | undefined): number | undefined {
  return typeof v === "number" ? v : undefined;
}

export interface GgufSummary {
  format: "gguf";
  name?: string;
  architecture?: string;
  parameters: string;
  quantization?: string;
  contextLength?: number;
  fileSize: string;
  tensorCount: number;
  metadataKeys: number;
  license?: string;
  hasChatTemplate: boolean;
}

export function summarizeGguf(m: GgufModel): GgufSummary {
  const md = m.metadata;
  const arch = typeof md["general.architecture"] === "string" ? (md["general.architecture"] as string) : undefined;
  const fileType = num(md["general.file_type"]);
  return {
    format: "gguf",
    name: typeof md["general.name"] === "string" ? (md["general.name"] as string) : undefined,
    architecture: arch,
    parameters: humanCount(m.paramCount),
    quantization: fileType !== undefined ? (FILE_TYPES[fileType] ?? `file_type_${fileType}`) : undefined,
    contextLength: arch ? num(md[`${arch}.context_length`]) : undefined,
    fileSize: humanBytes(m.fileSizeBytes),
    tensorCount: m.tensors.length,
    metadataKeys: Object.keys(md).length,
    license: typeof md["general.license"] === "string" ? (md["general.license"] as string) : undefined,
    hasChatTemplate: "tokenizer.chat_template" in md,
  };
}

export interface VramEstimate {
  weights: string;
  kvCache?: string;
  contextUsed?: number;
  total: string;
  note: string;
}

// KV cache: 2 (K and V) × layers × context × kv_heads × head_dim × 2 bytes (fp16).
// ponytail: assumes fp16 cache and ignores compute buffers; real usage runs a
// few percent higher — the note says so rather than faking precision.
export function estimateVram(m: GgufModel, contextLength?: number): VramEstimate {
  const md = m.metadata;
  const arch = typeof md["general.architecture"] === "string" ? (md["general.architecture"] as string) : undefined;
  const layers = arch ? num(md[`${arch}.block_count`]) : undefined;
  const heads = arch ? num(md[`${arch}.attention.head_count`]) : undefined;
  const kvHeads = (arch ? num(md[`${arch}.attention.head_count_kv`]) : undefined) ?? heads;
  const embed = arch ? num(md[`${arch}.embedding_length`]) : undefined;
  const headDim = (arch ? num(md[`${arch}.attention.key_length`]) : undefined) ?? (embed && heads ? embed / heads : undefined);
  const modelCtx = arch ? num(md[`${arch}.context_length`]) : undefined;

  if (!layers || !kvHeads || !headDim) {
    return {
      weights: humanBytes(m.weightsBytes),
      total: humanBytes(m.weightsBytes),
      note: "KV cache could not be estimated (architecture metadata missing); total covers weights only.",
    };
  }

  const ctx = Math.min(contextLength ?? Math.min(modelCtx ?? 4096, 8192), modelCtx ?? Infinity);
  const kvBytes = 2 * layers * ctx * kvHeads * headDim * 2;
  return {
    weights: humanBytes(m.weightsBytes),
    kvCache: humanBytes(kvBytes),
    contextUsed: ctx,
    total: humanBytes(m.weightsBytes + kvBytes),
    note: "Estimate assumes an fp16 KV cache and excludes compute buffers — expect a few percent more in practice.",
  };
}
