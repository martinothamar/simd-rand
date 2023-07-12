# Shishua for Rust

A Rust implementation of Shishua, 
[the worlds fastest PRNG](https://espadrine.github.io/blog/posts/shishua-the-fastest-prng-in-the-world.html) ([GitHub](https://github.com/espadrine/shishua)).

## Limitations

* Only AVX2 path
* Only x86_64

## Debugging/inspection

Requirements for disassembly:

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```

Then you can run `make dasm`

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

