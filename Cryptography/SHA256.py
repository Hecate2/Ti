# references:
# https://blog.boot.dev/cryptography/how-sha-2-works-step-by-step-sha-256/
# https://github.com/keanemind/python-sha-256/blob/master/sha256.py
from typing import Union
import taichi as ti

ti.init(arch=ti.gpu)

# read only
K = ti.field(dtype=ti.uint32, shape=64)
for i, k in enumerate([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]):
    K[i] = k


@ti.data_oriented
class Sha256:
    
    def __init__(self):
        # written for each hash, but not as state variables
        self.uint32_words = ti.field(dtype=ti.uint32, shape=512 // 32 + 48)  # 64 uint32 words
        self.s0_words = ti.field(dtype=ti.uint32, shape=16)
        self.s1_words = ti.field(dtype=ti.uint32, shape=16)
        self.abcdefgh = ti.field(dtype=ti.uint32, shape=8)
        self.abcdefgh2 = ti.field(dtype=ti.uint32, shape=8)

        # state variables
        self.hash_finished = False
        self.h = ti.field(dtype=ti.uint32, shape=8)
        for i, h_ in enumerate([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19, ]):
            self.h[i] = h_
        self.unhandled_bytes = bytearray()
        self.original_length_bits = 0
        # self.reset()
        
    def reset(self):
        self.hash_finished = False
        for i, h_ in enumerate([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19, ]):
            self.h[i] = h_
        self.unhandled_bytes = bytearray()
        self.original_length_bits = 0
        return self

    def update(self, b: Union[bytearray, bytes, str, memoryview]):
        assert self.hash_finished is False
        if type(b) is str:
            b: bytes = b.encode()
        if type(b) is bytes or type(b) is memoryview:
            b: bytearray = bytearray(b)
        original_length_b_bytes = len(b)
        self.original_length_bits += original_length_b_bytes * 8
        b: memoryview = memoryview(self.unhandled_bytes + b)
        
        blocks = (original_length_b_bytes + len(self.unhandled_bytes)) // (512 // 8)
        for _ in range(blocks):
            for i in range(512 // 32):
                self.uint32_words[i] = int.from_bytes(b[i * 4:i * 4 + 4], 'big', signed=False)
            self.hash()
            b = b[(512 // 8):]
        self.unhandled_bytes = bytearray(b)
        return self
    
    def finish(self, b: Union[bytearray, bytes, str, memoryview]):
        assert self.hash_finished is False
        self.hash_finished = True
        if type(b) is str:
            b: bytes = b.encode()
        if type(b) is bytes or type(b) is memoryview:
            b: bytearray = bytearray(b)
        original_length_bits = len(b) * 8 + self.original_length_bits  # len of bits
        b: bytearray = self.unhandled_bytes + b
        
        b.append(0x80)
        # now we want len_bits+64 is a multiple of 512
        # if not, we append 0x00 at the end of the input message
        length_bits = original_length_bits + 8  # in bits
        len_bits_append = ((length_bits + 64) % 512)  # in bits
        if len_bits_append != 0:
            len_bits_append = 512 - len_bits_append  # in bits
            b += b'\x00' * (len_bits_append // 8)
        b += original_length_bits.to_bytes(8, 'big')  # pad to 8 bytes or 64 bits
        length_bits = len(b)
        assert (length_bits * 8) % 512 == 0, "Padding error!"
        
        b: memoryview = memoryview(b)
        blocks = len(b) // (512 // 8)
        for _ in range(blocks):
            for i in range(512 // 32):
                self.uint32_words[i] = int.from_bytes(b[i * 4:i * 4 + 4], 'big', signed=False)
            self.hash()
            b = b[(512 // 8):]
        assert len(b) == 0, "Error at finish!"
        return self
    
    @ti.kernel
    def hash(self):
        for i in ti.static(range(16, 32)):
            self.s0_words[i - 16] = self._rotate_right(self.uint32_words[i - 15], 7) ^ self._rotate_right(self.uint32_words[i - 15], 18) ^ self._shift_right(self.uint32_words[i - 15], 3)
            self.s1_words[i - 16] = self._rotate_right(self.uint32_words[i - 2], 17) ^ self._rotate_right(self.uint32_words[i - 2], 19) ^ self._shift_right(self.uint32_words[i - 2], 10)
            self.uint32_words[i] = self.uint32_words[i - 16] + self.s0_words[i - 16] + self.uint32_words[i - 7] + self.s1_words[i - 16]
        for i in ti.static(range(32, 48)):
            self.s0_words[i - 32] = self._rotate_right(self.uint32_words[i - 15], 7) ^ self._rotate_right(self.uint32_words[i - 15], 18) ^ self._shift_right(self.uint32_words[i - 15], 3)
            self.s1_words[i - 32] = self._rotate_right(self.uint32_words[i - 2], 17) ^ self._rotate_right(self.uint32_words[i - 2], 19) ^ self._shift_right(self.uint32_words[i - 2], 10)
            self.uint32_words[i] = self.uint32_words[i - 16] + self.s0_words[i - 32] + self.uint32_words[i - 7] + self.s1_words[i - 32]
        for i in ti.static(range(48, 64)):
            self.s0_words[i - 48] = self._rotate_right(self.uint32_words[i - 15], 7) ^ self._rotate_right(self.uint32_words[i - 15], 18) ^ self._shift_right(self.uint32_words[i - 15], 3)
            self.s1_words[i - 48] = self._rotate_right(self.uint32_words[i - 2], 17) ^ self._rotate_right(self.uint32_words[i - 2], 19) ^ self._shift_right(self.uint32_words[i - 2], 10)
            self.uint32_words[i] = self.uint32_words[i - 16] + self.s0_words[i - 48] + self.uint32_words[i - 7] + self.s1_words[i - 48]
        for i in self.h:
            self.abcdefgh[i] = self.h[i]
        # abcdefgh
        # 01234567
        ti.loop_config(serialize=True)
        for t in range(32):
            # ping-pong operations
            temp1 = self.abcdefgh[7] + self._S1(self.abcdefgh[4]) + self._ch(self.abcdefgh[4], self.abcdefgh[5], self.abcdefgh[6]) + K[t * 2] + self.uint32_words[t * 2]
            temp2 = self._S0(self.abcdefgh[0]) + self._maj(self.abcdefgh[0], self.abcdefgh[1], self.abcdefgh[2])
            for i in ti.static(range(8)):
                if i == 0:
                    self.abcdefgh2[0] = temp1 + temp2
                elif i == 4:
                    self.abcdefgh2[4] = self.abcdefgh[3] + temp1
                else:
                    self.abcdefgh2[i] = self.abcdefgh[i - 1]
            temp1 = self.abcdefgh2[7] + self._S1(self.abcdefgh2[4]) + self._ch(self.abcdefgh2[4], self.abcdefgh2[5], self.abcdefgh2[6]) + K[t * 2 + 1] + self.uint32_words[t * 2 + 1]
            temp2 = self._S0(self.abcdefgh2[0]) + self._maj(self.abcdefgh2[0], self.abcdefgh2[1], self.abcdefgh2[2])
            for i in ti.static(range(8)):
                if i == 0:
                    self.abcdefgh[0] = temp1 + temp2
                elif i == 4:
                    self.abcdefgh[4] = self.abcdefgh2[3] + temp1
                else:
                    self.abcdefgh[i] = self.abcdefgh2[i - 1]
        for i in ti.static(range(8)):
            self.h[i] = self.h[i] + self.abcdefgh[i]
    
    @ti.func
    def _S0(self, a: ti.uint32):
        return self._rotate_right(a, 2) ^ self._rotate_right(a, 13) ^ self._rotate_right(a, 22)
    
    @ti.func
    def _S1(self, e: ti.uint32):
        return self._rotate_right(e, 6) ^ self._rotate_right(e, 11) ^ self._rotate_right(e, 25)
    
    @staticmethod
    @ti.func
    def _ch(e: ti.uint32, f: ti.uint32, g: ti.uint32):
        return (e & f) ^ (~e & g)
    
    @staticmethod
    @ti.func
    def _maj(a: ti.uint32, b: ti.uint32, c: ti.uint32):
        return (a & b) ^ (a & c) ^ (b & c)
    
    @staticmethod
    @ti.func
    def _rotate_right(num: ti.uint32, shift: ti.uint32, size: ti.uint8 = 32):
        return (num >> shift) | (num << (size - shift))
    
    @staticmethod
    @ti.func
    def _shift_right(i: ti.uint32, shift: ti.uint8) -> ti.uint32:
        return i >> shift


if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(formatter={'int': hex})
    s = Sha256()
    print(s.reset().finish(b'hello world').h.to_numpy())  # b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    print(s.reset().finish(b'a').h.to_numpy())  # 0xca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb
    print(s.reset().finish(b'ab').h.to_numpy())  # 0xfb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603
    print(s.reset().update(b'ab').finish(b'').h.to_numpy())  # 0xfb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603
    # assert Sha256().finish(b'a') == 0xca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb
    # assert Sha256().finish(b'ab') == 0xfb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603
    print(s.reset().finish(b'a'*720).h.to_numpy())  # 7ea11a437a3eedb3bb17509cc5cf05529eb0141a6ac9b4f98243943d065eb00d
    print(s.reset().update(b'a'*700).update(b'a'*4).update(b'a'*16).finish(b'').h.to_numpy())  # 7ea11a437a3eedb3bb17509cc5cf05529eb0141a6ac9b4f98243943d065eb00d
    
    import time
    start_time = time.time()
    for _ in range(100):
        s.reset().update(b'a' * 700).update(b'a' * 4).update(b'a' * 16).finish(b'')
    print(f'{time.time() - start_time} seconds for 100 serialized sha256 with gpu')

    from hashlib import sha256
    start_time = time.time()
    for _ in range(1000000):
        sha256(b'a'*720)
    print(f'{time.time() - start_time} seconds for 1000000 sha256 with hashlib')
    # ti.profiler.print_scoped_profiler_info()
