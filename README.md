# hpc-project

PC specs:

Intel Core i5-10400F

MSI NVIDIA GeForce RTX 3060 VENTUS 2X OC

16Gb 3200Mhz DDR4 RAM

# Task 1: Matrix Multiplication

![alt text](https://user-images.githubusercontent.com/52837578/148709747-17dbab23-37a3-43ec-81a2-53c08465497d.png)

As we can see on this image gpu calculations faster that cpu calculations, as it was expected. This is explained by the fact that the algorithm on the GPU is parallelized, which significantly speeds up the calculation process, while on the CPU this algorithm is executed sequentially, which, with a significant increase in dimension, critically affects the speed of calculations.

# Task 2: Vector Sum

![alt text](https://user-images.githubusercontent.com/52837578/148709964-5719439d-2ad2-48f4-a262-475e297ea56b.png)

On this image we see lots of spices, so it's imposible to say what type of calculations is better. So let's take a look on separate graphics.

![alt text](https://user-images.githubusercontent.com/52837578/148710042-285430b0-f740-497c-b1f0-e1a5426af724.png)

From the picture above we can see that usage of GPU is not always justified. In part, this is due to the relatively small size of the array and the considerable time spent on parallelizing the algorithm (including the synchronization process). However, with the increase in the dimension of the array, the use of the GPU becomes justified, since the time spent on parallelization and calculations performed on the GPU becomes less compared to the time spent by the CPU on the operation of the sequential algorithm.

# Task 3: Salt And Pepper Noise
Base image resolution is: 2560x1600

![alt text](https://user-images.githubusercontent.com/52837578/148710405-5ed55eab-8c99-4bc3-ad99-4456c8c458f2.png)
Gray scale image
![alt text](https://user-images.githubusercontent.com/52837578/148710417-4debea2d-e290-4b8a-9f50-261b7cd34b5f.png)
Gray scale image + S&P noise from skimage library
![alt text](https://user-images.githubusercontent.com/52837578/148710433-00407bad-5c9b-4712-864b-0ebf95d7123e.png)
Gray scale image + CPU noise
![alt text](https://user-images.githubusercontent.com/52837578/148710442-14304913-b6d5-4843-98af-6f2bb4c7f22a.png)
Gray scale image + GPU noise


CPU Time:  10.194026947021484

GPU Time:  0.34554338455200195

Based on the temporal performance of the algorithms, as well as the acceleration rate,
it becomes obvious that the use of GPU for filtering is completely justified.

# Task 4: Pi Number Calculations
![alt text](https://user-images.githubusercontent.com/52837578/148710675-5f16d6c7-5da6-4ae6-8665-a301f1942e79.png)


     N    gpu_time    cpu_time    gpu_pi    cpu_pi    acceleration
------  ----------  ----------  --------  --------  --------------
 10000  0.979175      0.2339     3.14015   3.13998        0.238874
 20000  0.00497985    0.466735   3.14141   3.14324       93.7248
 30000  0.00897622    0.702133   3.14132   3.14079       78.2215
 40000  0.011004      0.916514   3.14367   3.14197       83.2894
 50000  0.048898      1.15414    3.14333   3.1414        23.6029
 60000  0.0618396     1.39305    3.14254   3.14472       22.5269
 70000  0.0720167     1.62454    3.14225   3.14162       22.5578
 80000  0.0809202     1.85376    3.14227   3.14192       22.9085
 90000  0.0887427     2.12514    3.14268   3.14091       23.9472
100000  0.105108      2.32553    3.14336   3.14051       22.1252



As we can see, algorithm for calculating the number of PI on the CPU and GPU was implemented.
Both algorithms show good accuracy of calculations, and it is also shown that the algorithm implemented
on the GPU makes it faster. The maximum acceleration obtained as a result of the experiment was almost 100 times faster.

