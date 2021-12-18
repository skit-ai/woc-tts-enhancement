## TestSet

This testset is created by sampling from the [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/) dataset. The noisy database contains 30 IEEE sentences (produced by three male and three female speakers) corrupted by eight different real-world noises at different SNRs. The noise includes suburban train noise, babble, car, exhibition hall, restaurant, street, airport and train-station noise. Our testset is created by randomly sampling from the distorted speech samples and keeping the clean speeches intact.

The clean subdirectory has 30 files while the noisy subdirectory contains 102 files which can be used to test out the different methods of noise reduction.