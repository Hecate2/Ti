from multiprocessing import cpu_count
cpu_count = cpu_count()
import time
from typing import Dict, Tuple
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, device_memory_GB=3)

num_digits = 4
num_numbers = 10 ** num_digits

valid_guesses = ti.field(dtype=ti.uint8, shape=num_numbers)
digits = ti.field(dtype=ti.uint8, shape=(num_numbers, num_digits))
"""
digits = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],  # num 0001
    ...
    [4, 3, 2, 1],  # num 1234
    ...
]
"""
guess_to_answer_A_and_B_sum = ti.field(dtype=ti.uint8, shape=(num_numbers, num_numbers, 2))
remaining_guesses_if_guess_this = ti.field(dtype=ti.uint32, shape=num_numbers)
A_and_B = ti.field(dtype=ti.uint8, shape=(10 ** num_digits, 2))


@ti.func
def get_digit(num, digit_index: int) -> int:
    """
    :param num: 65535
    :param digit_index: 0
    :return: 5
    """
    return ti.cast((num // 10 ** digit_index) % 10, ti.uint8)


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
                valid_guesses[num] = ti.cast(0, ti.uint8)
                # break  # WATCH THIS!
                # The `break` might have been compiled
                # outside this `if` clause


@ti.func
def compute_A_and_B():
    guess_to_answer_A_and_B_sum.fill(0)
    for solution in range(num_numbers):
        if valid_guesses[solution] != 0:
            ti.loop_config(block_dim=128, parallelize=cpu_count)
            for guess in range(num_numbers):
                if valid_guesses[guess] != 0:
                    for d_guess in ti.static(range(num_digits)):
                        vectorA = ti.Vector([0] * num_digits, dt=ti.uint8)
                        vectorB = ti.Vector([0] * num_digits, dt=ti.uint8)
                        if digits[solution, d_guess] == digits[guess, d_guess]:
                            vectorA[d_guess] = ti.cast(1, ti.uint8)
                        for d_solution in ti.static(range(num_digits)):
                            if d_solution != d_guess and digits[solution, d_solution] == digits[guess, d_guess]:
                                vectorB[d_guess] = ti.cast(1, ti.uint8)
                        sum_vectorA = vectorA.sum()
                        guess_to_answer_A_and_B_sum[solution, guess, 0] = sum_vectorA
                        guess_to_answer_A_and_B_sum[guess, solution, 0] = sum_vectorA
                        sum_vectorB = vectorB.sum()
                        guess_to_answer_A_and_B_sum[solution, guess, 1] = sum_vectorB
                        guess_to_answer_A_and_B_sum[guess, solution, 1] = sum_vectorB


@ti.kernel
def initialize_one_game():
    valid_guesses.fill(1)
    no_duplicate_digits()


@ti.kernel
def initialize():
    get_digits()
    compute_A_and_B()
    valid_guesses.fill(1)
    no_duplicate_digits()


@ti.kernel
def reduce_possible_guesses(guess: ti.uint32, actual_A: ti.uint8, actual_B: ti.uint8):
    """
    :param guess: guessed number. e.g. 2043
    :param actual_A: num of same digit in the answer at correct position
    :param actual_B: num of same digit in the answer but not at correct position
    """
    valid_guesses[guess] = 0
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



@ti.kernel
def find_best_guess():
    """
    :return: the expected amount of information that will be generated by each guess
    """
    remaining_guesses_if_guess_this.fill(0)
    for actual_answer in range(num_numbers):
        if valid_guesses[actual_answer] == 0:
            remaining_guesses_if_guess_this[actual_answer] = ti.cast(2 ** 31 - 1, ti.uint32)
        else:
            ti.loop_config(block_dim=128, parallelize=cpu_count)
            for next_guess in range(num_numbers):
                for a in ti.static(range(num_digits + 1)):
                    for b in ti.static(range(num_digits + 1 - a)):
                        if guess_to_answer_A_and_B_sum[actual_answer, next_guess, 0] == a and guess_to_answer_A_and_B_sum[actual_answer, next_guess, 0] == b:
                            remaining_guesses_if_guess_this[next_guess] += 1


initialize()
# ti.profiler.print_scoped_profiler_info()
# print(digits)
# print(initial_nums)
# print(valid_guesses)
# result_guesses = np.where(valid_guesses.to_numpy() != 0)[0]
# print(result_guesses)


def play_once():
    while (valid_guesses_shape := np.where(valid_guesses.to_numpy() != 0)[0].shape[0]) > 0:
        find_best_guess()
        sorted_suggested_guesses = remaining_guesses_if_guess_this.to_numpy().argsort()
        suggested_guess = int(sorted_suggested_guesses[0])
        print(valid_guesses_shape)
        print(str(sorted_suggested_guesses[0]).zfill(num_digits), end='\t')
        A, B = map(int, input().split())
        reduce_possible_guesses(suggested_guess, A, B)


def test():
    print('You are testing the performance of this program')
    print('Call play_once() instead of test() to play 1A2B with IntelligentIthea using the best strategy')
    print('1000 simulations cost around 15 seconds on RTX 2070 at 80 Watts')
    print('Please wait for simulation...')
    import random
    
    def gen_solution() -> Dict[int, int]:
        digits = random.sample("0123456789", num_digits)  # ['3','6','2','0']
        return {int(d): num_digits - i - 1 for i, d in enumerate(digits)}  # {3:3, 6:2, 2:1, 0:0}
    
    def gen_A_B(solution: Dict[int, int], guess: int) -> Tuple[int, int]:
        guess_dict = dict()
        for i in range(num_digits):
            guess_dict[i] = guess % 10
            guess = guess // 10
        A, B = 0, 0
        for k, v in guess_dict.items():
            if k in solution:
                if solution[k] == v:
                    A += 1
                else:
                    B += 1
        return A, B
    
    initialize()
    total_guesses = 0
    max_guesses = 0
    min_guesses = 100
    tries = 1000
    start_time = time.time()
    for i in range(tries):
        game_guesses = 0
        solution = gen_solution()
        initialize_one_game()
        while (np.where(valid_guesses.to_numpy() != 0)[0].shape[0]) > 0:
            find_best_guess()
            sorted_suggested_guesses = remaining_guesses_if_guess_this.to_numpy().argsort()
            suggested_guess = int(sorted_suggested_guesses[0])
            total_guesses += 1
            game_guesses += 1
            A, B = gen_A_B(solution, suggested_guess)
            reduce_possible_guesses(suggested_guess, A, B)
        if game_guesses < min_guesses:
            min_guesses = game_guesses
        if game_guesses > max_guesses:
            max_guesses = game_guesses
    end_time = time.time()
    print(f'time cost: {end_time - start_time} seconds with {tries} tries')
    print(f'min: {min_guesses}, avg: {total_guesses/tries}, max: {max_guesses}')


# play_once()
test()