import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

num_digits = 4

valid_guesses = ti.field(dtype=ti.uint8, shape=10 ** num_digits)
digits = ti.field(dtype=ti.uint8, shape=(10 ** num_digits, num_digits))
"""
digits = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],  # num 0001
    ...
    [4, 3, 2, 1],  # num 1234
    ...
]
"""
A_and_B = ti.field(dtype=ti.uint8, shape=(10 ** num_digits, 2))


@ti.func
def get_digit(num, digit_index: int) -> int:
    """
    :param num: 65535
    :param digit_index: 0
    :return: 5
    """
    return (num // 10 ** digit_index) % 10


@ti.func
def get_digits():
    for num, d in digits:
        digits[num, d] = get_digit(num, d)


@ti.func
def no_duplicate_digits():
    for num, d in digits:
        for j in ti.static(range(num_digits)):
            # if num == 1:
            #     print(num, d, j, digits[num, d] == digits[num, j])
            if d != j and digits[num, d] == digits[num, j]:
                valid_guesses[num] = 0
                # break  # WATCH THIS!
                # The `break` might have been compiled
                # outside this `if` clause


@ti.kernel
def get_initial_nums():
    valid_guesses.fill(1)
    get_digits()
    no_duplicate_digits()


@ti.kernel
def reduce_possible_guesses(guess: ti.uint32, actual_A: ti.uint8, actual_B: ti.uint8):
    """
    :param guess: guessed number. e.g. 2043
    :param actual_A: num of same digit in the answer at correct position
    :param actual_B: num of same digit in the answer but not at correct position
    """
    for num in valid_guesses:
        if valid_guesses[num] == 1:
            A_and_B[num, 0] = 0
            A_and_B[num, 1] = 0
            vectorA = ti.Vector([0] * num_digits)
            vectorB = ti.Vector([0] * num_digits)
            for d in ti.static(range(num_digits)):
                if digits[num, d] == digits[guess, d]:
                    vectorA[d] = 1
                for d_inner in ti.static(range(num_digits)):
                    if d != d_inner and digits[num, d] == digits[guess, d_inner]:
                        vectorB[d_inner] = 1
            A_and_B[num, 0] = vectorA.sum()
            A_and_B[num, 1] = vectorB.sum()
            if A_and_B[num, 0] != actual_A or A_and_B[num, 1] != actual_B:
                valid_guesses[num] = 0


get_initial_nums()
# print(digits)
# print(initial_nums)
# print(valid_guesses)
result_guesses = np.where(valid_guesses.to_numpy() != 0)[0]
while result_guesses.shape[0] > 1:
    print(str(result_guesses[0]).zfill(num_digits), end='\t')
    A, B = map(int, input().split())
    reduce_possible_guesses(int(result_guesses[0]), A, B)
    result_guesses = np.where(valid_guesses.to_numpy() != 0)[0]
    print(result_guesses.shape)
    print(result_guesses)
ti.profiler.print_scoped_profiler_info()