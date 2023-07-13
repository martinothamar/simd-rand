# Shishua for Rust

A Rust implementation of Shishua, 
['the worlds fastest PRNG'](https://espadrine.github.io/blog/posts/shishua-the-fastest-prng-in-the-world.html) ([GitHub](https://github.com/espadrine/shishua)).

## Limitations

* Only AVX2 path (x86_64)

## Performance

### Benchmarks

#### Word sampling (`next_u64`)

Each iterations extracts 64 u64 values in a loop. This equates to 512 bytes per iteration.

```
    Finished bench [optimized + debuginfo] target(s) in 0.04s
     Running benches/rng.rs (target/release/deps/rng-a5d22c2c623a41ec)
Benchmarking RNG/Shishua u64
Benchmarking RNG/Shishua u64: Warming up for 3.0000 s
Benchmarking RNG/Shishua u64: Collecting 100 samples in estimated 5.0001 s (118336650 iterations)
Benchmarking RNG/Shishua u64: Analyzing
RNG/Shishua u64         time:   [42.041 ns 42.134 ns 42.247 ns]
Found 8 outliers among 100 measurements (8.00%)
  5 (5.00%) high mild
  3 (3.00%) high severe
slope  [42.041 ns 42.247 ns] R^2            [0.9879274 0.9876047]
mean   [42.074 ns 42.278 ns] std. dev.      [331.70 ps 722.91 ps]
median [42.038 ns 42.163 ns] med. abs. dev. [214.35 ps 375.16 ps]
------------------------------------------------------------------------------------------------
Benchmarking RNG/SmallRng u64
Benchmarking RNG/SmallRng u64: Warming up for 3.0000 s
Benchmarking RNG/SmallRng u64: Collecting 100 samples in estimated 5.0001 s (113064450 iterations)
Benchmarking RNG/SmallRng u64: Analyzing
RNG/SmallRng u64        time:   [44.710 ns 44.927 ns 45.162 ns]
Found 7 outliers among 100 measurements (7.00%)
  5 (5.00%) high mild
  2 (2.00%) high severe
slope  [44.710 ns 45.162 ns] R^2            [0.9477585 0.9471898]
mean   [44.635 ns 44.964 ns] std. dev.      [654.10 ps 1.0341 ns]
median [44.429 ns 44.693 ns] med. abs. dev. [389.24 ps 703.26 ps]
```

### Assembly

Example disassembly output

<table>
<tr>
<th>Shishua</th>
<th>SmallRng</th>
</tr>
<tr>
<td>
  
```asm
00000000000066b0 <dasm::do_shishua::hcb7630cbc5ecd283>:
; fn do_shishua(rng: &mut Shishua) -> u64 {
    66b0:      	push	rbx
;         if BUFFER_SIZE - self.buffer_index < size {
    66b1:      	mov	rcx, qword ptr [rdi + 262432]
    66b8:      	mov	rbx, rdi
    66bb:      	lea	rax, [rcx - 262137]
    66c2:      	cmp	rax, 8
    66c6:      	jb	0x66d9 <dasm::do_shishua::hcb7630cbc5ecd283+0x29>
    66c8:      	mov	rax, qword ptr [rbx + rcx]
;             state.buffer_index += N;
    66cc:      	add	rcx, 8
    66d0:      	mov	qword ptr [rbx + 262432], rcx
; }
    66d7:      	pop	rbx
    66d8:      	ret
;             self.rebuffer();
    66d9:      	mov	rdi, rbx
    66dc:      	call	0x4050 <shishua::BufferedState<_>::rebuffer::h90eab48a07bd5f7f>
;                 .get_unchecked(state.buffer_index..state.buffer_index + N);
    66e1:      	mov	rcx, qword ptr [rbx + 262432]
    66e8:      	jmp	0x66c8 <dasm::do_shishua::hcb7630cbc5ecd283+0x18>
    66ea:      	nop	word ptr [rax + rax]
```
  
</td>
<td>

```asm
00000000000066f0 <dasm::do_small_rng::h0abe4afe95d09cce>:
;             .wrapping_add(self.s[3])
    66f0:      	mov	r9, qword ptr [rdi + 24]
;         let result_plusplus = self.s[0]
    66f4:      	mov	rdx, qword ptr [rdi]
    66f7:      	mov	rcx, qword ptr [rdi + 16]
;         let t = self.s[1] << 17;
    66fb:      	mov	rsi, qword ptr [rdi + 8]
    66ff:      	lea	rax, [r9 + rdx]
;         self.s[2] ^= self.s[0];
    6703:      	xor	rcx, rdx
;         self.s[3] ^= self.s[1];
    6706:      	xor	r9, rsi
;         let t = self.s[1] << 17;
    6709:      	mov	r8, rsi
    670c:      	shl	r8, 17
    6710:      	rorx	rax, rax, 41
;         self.s[1] ^= self.s[2];
    6716:      	xor	rsi, rcx
;         self.s[2] ^= t;
    6719:      	xor	rcx, r8
    671c:      	add	rax, rdx
;         self.s[0] ^= self.s[3];
    671f:      	xor	rdx, r9
;         self.s[1] ^= self.s[2];
    6722:      	mov	qword ptr [rdi + 8], rsi
;         self.s[0] ^= self.s[3];
    6726:      	mov	qword ptr [rdi], rdx
    6729:      	rorx	rdx, r9, 19
;         self.s[2] ^= t;
    672f:      	mov	qword ptr [rdi + 16], rcx
;         self.s[3] = self.s[3].rotate_left(45);
    6733:      	mov	qword ptr [rdi + 24], rdx
; }
    6737:      	ret
    6738:      	nop	dword ptr [rax + rax]
```

</td>
</tr>
</table>

### Notes

Requirements for disassembly:

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```

Then you can run `make dasm`
