#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <appMacros.h>
#include <unistd.h>
#include <stdint.h>

void printUsage(char* appName);
int parseArgs(char** argv,int* pSize,int *print,int argc);
uint64_t getTime();

__global__ void prime(int *a, int count) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	//Handle the data at the index
	if(tid > count) return;

	int can = a[tid];
	int counter=3;
	//int flag=0; 
	//float limit = sqrtf((float)can);
	float limit = sqrtf(can);
	limit = limit+1;
	// if even -- get out
	if(can%2==0)
	{
		a[tid] = 1;
	}
	else
	{	
		for(;counter<=limit;counter+=2)
		{
			if(can%counter==0)
			{
				a[tid]=1; // set as prime
				break;
			}
		}
	}
}



int main(int argc, char** argv)
{
	/**
	  Check the number of arguments 
		1. ./app probsize
		2. ./app -t probsize
	**/

	if(argc < 2)
	{
		printUsage(argv[ARGS_APP_NAME]);
		return FAILURE;
	}


	int print=FALSE;
	int problemSize=0;

	// parse the arguments to check for 
	parseArgs(argv,&problemSize,&print,argc);
	
	if(problemSize == 0)
	{
		printf("Please enter a proper number for problem size\n");
		printUsage(argv[ARGS_APP_NAME]);
		return FAILURE;
	}

	int *candidates;
	int *dev_candidates;
	int i;
	candidates = (int*)calloc(problemSize,sizeof(int));
	cudaMalloc((void**)&dev_candidates, problemSize * sizeof (int) );
	
	//fill the arrays 'a' and 'b' on the CPU
	for ( i = 0; i < problemSize; i++) 
	{
		candidates[i]=i+1;
	}
	
	int threads_IN_block = THREADS_PER_BLOCK;
	int blocks = (problemSize/threads_IN_block)+1;


	//copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_candidates,candidates,problemSize * sizeof(int),cudaMemcpyHostToDevice);
	double start, end;
	start = getTime();
	prime<<<blocks, threads_IN_block>>> (dev_candidates,problemSize);
	end = getTime();
	cudaMemcpy(candidates, dev_candidates, problemSize * sizeof(int),cudaMemcpyDeviceToHost );

					
	//display the resultsn
	
	if(print == true) // only print if requested
	{
		printf("============= PRIME LIST ================\n");
		for (i=0;i<problemSize;i++) 
		{
			if(candidates[i]!=1) // means this is not prime
			{
				printf("%d\n",i+1);		
			}
			
		}
		printf("\n");	
	}
	//else // give count of prime numbers and al
	{
		int primesCount = 0;
		int largestPrime = 0;
		for (i=0;i<problemSize;i++) 
		{
			if(candidates[i]!=1) 
			{
				//printf("%d is prime\n",i+1);	
				largestPrime=i+1;	
				primesCount++;
			}
			
		}
		printf("============= summary ================\n");
		printf("The number of primes is %d\n",primesCount);
		printf("The largest prime number is %d\n",largestPrime);
		
	}
	

	printf("Time for finding prime numbers till %d is %f seconds. \n",problemSize,(end-start)/1000000.0);
	
	//free the memory allocated on the GPU
	cudaFree(dev_candidates);
	return 0;
}



void printUsage(char* appName)
{
	printf("---------------- Wrong number of arguments ---------------\n");
	printf("%s <PROBLEM_SIZE>\n",appName);
	printf("%s <-t PROBLEM_SIZE> \n",appName);
	printf("---------------- Wrong number of arguments ---------------\n");
}

int parseArgs(char** argv,int* problemSize,int* print,int arg_count)
{
	if(arg_count == 3)
	{
		if(!strcmp(argv[ARGS_P_SIZE_ID],ARGS_P_SIZE_ID_CODE))
		{
			*print=TRUE;
			*problemSize=atoi(argv[ARGS_P_SIZE]);
		}
	}
	else if(arg_count == 2)
	{
		*problemSize =atoi(argv[1]);
	}
	return SUCCESS;
}

uint64_t getTime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (uint64_t)(t.tv_sec)*1000000 + (uint64_t)(t.tv_usec);
}
