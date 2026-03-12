# jit-whisperer

> Taming the JIT — low-level C# patterns for SIMD, FFI, and cross-language interop.

## What is this

A collection of production-grade C# techniques that operate at the boundary between
managed .NET and native execution: SIMD intrinsics, memory layout control, JIT hints,
and FFI wiring to external systems such as MongoDB.

The code in this repository is intentionally close to the metal.

## Covered Patterns

- **SIMD pruning** — vectorized range checks via `System.Numerics.Vector<T>`
- **Memory layout** — `[StructLayout]`, padding, and ABI-safe struct design  
- **JIT guidance** — `[MethodImpl(AggressiveInlining)]` and when it actually matters
- **Morton encoding** — Z-order curves for spatial locality in 11-dimensional affine space
- **MongoDB interop** — BRIN-style B\*-Tree index driving targeted range queries

## Requirements

- .NET 5 SDK 以上
- MongoDB 6+ (local or Atlas)

## Build

```bash
dotnet build
```

The repository now includes a project file, so building from the repository root works directly.
