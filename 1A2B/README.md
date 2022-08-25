Thanks to [`bsTiat`](https://www.luogu.com.cn/user/211086) (also known as `bluespace`) and [`xgnd`](https://github.com/xgnd/) (a major contributor of the QQ bot [IntelligentIthea](https://github.com/xgnd/IntelligentIthea)).

1A2B is a game also known as Bulls and Cows. A 4-digit number without duplicating digit is chosen by the host player, and it is the guesser player's job to guess what the number is. In our version, it is allowed to choose a number starting with a single digit 0. 

Whenever the guesser guesses a number, the host should answer a string `aAbB`, where `a` and `b` are numbers. `aA` means that the guessed number has `a` digits at the same position of the correct solution. `bB` means that the guessed number has `b` digits included by the solution, but not in the correct position. For example, for the solution `8317`, we may have the following guessing process:

```
0123	0A2B
1045	0A1B
2356	1A0B
2708	0A2B
7381	1A3B
[8317]	4A0B (correct answer!)
```

The program [1a2b_gpu.py](1a2b_gpu.py) helps the guesser filter the impossible answers with GPU. It prints the answer to be provided to the host, and it's your job to input `a` and `b` back to the program. (No need to input `aAbB`. You can just type `a b`.) Then the program will print all the remaining possible solutions, and (naively) select the smallest number as the next answer. 

```
Rhantolk@HISTORIA C:\Users\RhantolkYtriHistoria
# "C:\Program Files\Python38\python.exe" C:/Users/RhantolkYtriHistoria/Ti/1A2B/1a2b_gpu.py
[Taichi] version 1.1.2, llvm 10.0.0, commit f25cf4a2, win, python 3.8.10
[Taichi] Starting on arch=cuda
0123	0 2
(1260,)
[1045 1046 1047 ... 9830 9831 9832]
1045	0 1
(320,)
[2356 2357 2358 2359 2364 2374 2384 2394 2436 2437 2438 2439 2536 2537
 2538 2539 2607 2608 2609 2617 2618 2619 2634 2670 2671 2680 2681 2690
 2691 2706 2708 2709 2716 2718 2719 2734 2760 2761 2780 2781 2790 2791
 2806 2807 2809 2816 2817 2819 2834 2860 2861 2870 2871 2890 2891 2906
 2907 2908 2916 2917 2918 2934 2960 2961 2970 2971 2980 2981 3256 3257
 3258 3259 3264 3274 3284 3294 3462 3472 3482 3492 3562 3572 3582 3592
 3607 3608 3609 3617 3618 3619 3652 3670 3671 3680 3681 3690 3691 3706
 3708 3709 3716 3718 3719 3752 3760 3761 3780 3781 3790 3791 3806 3807
 3809 3816 3817 3819 3852 3860 3861 3870 3871 3890 3891 3906 3907 3908
 3916 3917 3918 3952 3960 3961 3970 3971 3980 3981 4236 4237 4238 4239
 4362 4372 4382 4392 4632 4732 4832 4932 5236 5237 5238 5239 5362 5372
 5382 5392 5632 5732 5832 5932 6207 6208 6209 6217 6218 6219 6234 6270
 6271 6280 6281 6290 6291 6307 6308 6309 6317 6318 6319 6352 6370 6371
 6380 6381 6390 6391 6432 6532 6702 6712 6730 6731 6802 6812 6830 6831
 6902 6912 6930 6931 7206 7208 7209 7216 7218 7219 7234 7260 7261 7280
 7281 7290 7291 7306 7308 7309 7316 7318 7319 7352 7360 7361 7380 7381
 7390 7391 7432 7532 7602 7612 7630 7631 7802 7812 7830 7831 7902 7912
 7930 7931 8206 8207 8209 8216 8217 8219 8234 8260 8261 8270 8271 8290
 8291 8306 8307 8309 8316 8317 8319 8352 8360 8361 8370 8371 8390 8391
 8432 8532 8602 8612 8630 8631 8702 8712 8730 8731 8902 8912 8930 8931
 9206 9207 9208 9216 9217 9218 9234 9260 9261 9270 9271 9280 9281 9306
 9307 9308 9316 9317 9318 9352 9360 9361 9370 9371 9380 9381 9432 9532
 9602 9612 9630 9631 9702 9712 9730 9731 9802 9812 9830 9831]
2356	1 0
(48,)
[2708 2709 2718 2719 2780 2781 2790 2791 2807 2809 2817 2819 2870 2871
 2890 2891 2907 2908 2917 2918 2970 2971 2980 2981 7308 7309 7318 7319
 7380 7381 7390 7391 8307 8309 8317 8319 8370 8371 8390 8391 9307 9308
 9317 9318 9370 9371 9380 9381]
2708	0 2
(7,)
[7381 7390 8317 8371 8390 9370 9380]
7381	1 3
(1,)
[8317]
```

We have also written [1a2b_gpu_best_strategy.py](1a2b_gpu_best_strategy.py) which chooses a number generating the largest expected amount of information. `bsTiat` (also known as `bluespace`) has already tried the strategy with CPU, earning a small but significant benefit. 

