/*=============================================================================
* File Name: primGen.c
 * Project  : PDP Assignment 1
 * Version  : 0.1V
 * Author   : Rajaram Rabindranath (50097815)
 * Created  : November 30 2014
 ============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <appMacros.h>
#include <math.h>
#include <time.h>
#include <lib.h>
#include <sys/time.h>
int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printUsage(argv[ARGS_APP_NAME]);
		return FAILURE;
	}

	int print=FALSE;
	int problemSize=0;
	parseArgs(argv,&problemSize,&print,argc);
	int bucketSize=0;

	// if non-number is given as argument
	if(problemSize == 0)
	{
		printf("Please enter a proper number for problem size\n");
		printUsage(argv[ARGS_APP_NAME]);
		return FAILURE;
	}

	/** time the run **/
	struct timeval begin, end;
	int isPrime=TRUE; // a number is a candidate prime until proven otherwise
	gettimeofday(&begin,NULL);
	for(int i=3;i<=problemSize;i+=2)
	{
		for(int j=3;j<sqrt(i);j+=2)
		{
			if(i%j==0)
			{
				isPrime=FALSE;
				break;
			}
		}
		if(isPrime && print == TRUE)
		{
			printf("%d\n",i);
		}
		isPrime=TRUE;

	}
	gettimeofday(&end,NULL);
	double TimeElapsed =(end.tv_sec - begin.tv_sec) +((end.tv_usec - begin.tv_usec)/1000000.0);
	printf("Total time taken in secs to find all primes till %d is %f\n",problemSize,TimeElapsed);
}

