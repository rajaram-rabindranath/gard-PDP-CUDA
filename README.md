FOLDER STRUCTURE:
Prime 	-- Contains GPU and SEQ codes for Prime number generation
GPU/	Sequential/
Sort	-- Contains GPU and SEQ codes for bucket Sorting
GPU/ Sequential/


HOW - TO:

VERY IMPOTANT NOTICE:

Sorting:

When running the program -- please make sure that 
	FOR Sequential code -- problem size has to be divisible by 5
	For GPU code -- Please do enter a number that is a multiple of 1000 like say 10,000


Note all the results are directed to be stored on the "result" folder provided for each sub_directory

Inside each of the aforementioned folders is a runner script to submit the program 
in SBATCH and here are their names, what they do and how to run them.

------------------------------------------------ Prime/Sequential FOLDER ---------------------------------------------------
runner_prime_SEQ.sh -- the runner script for Sequential code
e.g
	$ ./runner_prime_SEQ.sh


The runner file has to be modified to change problem size as follows
(The things to be modified are right at the top of the file and have FIXME tags)

// to set the problem size fill the primeLimits array like so 
### FIXME --- Here is where one can set the problem size --- like so --- probSize(20 30 40) below is an example
probSize=(100000000);

--- to run in non batch and to execute on local machine do the following:

	./primeSEQ <-t probSize>  --- to print output + time
	./primeSEQ <probSize>  --- to print output + time

 $ make
 $ ./primeSEQ -t 45


----------------------------------------------Prime/GPU------------------------------------------------------
runner_GPU_Prime.sh -- the runner script for MPI and requires an argument for Nodes
e.g

The runner file has to be modified to change problem size and the number of cores as follows
The things to be modified are right at the top of the file and have FIXME tags

// to set the problem size fill the primeLimits array like so 
## FIXME here is where one can set probSise--- probSize=(24)
probSize=(25)


--- to run in non batch and to execute local machine do the following:

	# to print output + time
	$ ./cudaPrime <-t probSize> 
	# only print time
	$ ./cudaPrime <probSize> 
 
 $ make
 $./cudaPrime 45


------------------------------------------------Sort/Sequential---------------------------------------------------
runner_sort_SEQ.sh -- the runner script for OMP

	$ ./runner_sort_SEQ.sh

The runner file has to be modified to change problem size
The things to be modified are right at the top of the file and have FIXME tags

// to set the problem size fill the primeLimits array like so 
## FIXME here is where one can set probSize like so --- probSize=(24)
probSize=(25)



--- to run in non batch and to execute local machine do the following:

	# to print output + time
	$ ./bucketSort <-t probSize> 

	# to print just time
	$ ./bucketSort <probSize> 


 $ make
 $ ./bucketSort -t 45


------------------------------------------------Sort/GPU---------------------------------------------------
runner_gpu_sort.sh -- the runner script for OMP

	$ ./runner_gpu_sort.sh

The runner file has to be modified to change problem size
The things to be modified are right at the top of the file and have FIXME tags

// to set the problem size fill the primeLimits array like so 
## FIXME here is where one can set probSize like so --- probSize=(24)
probSize=(25)

--- to run in non batch and to execute local machine do the following:

	# to print output + time
	$ ./cudaBSORT <-t probSize> 

	# to print just time
	$ ./cudaBSORT <probSize> 


 $ make
 $ ./cudaBSORT -t 45




