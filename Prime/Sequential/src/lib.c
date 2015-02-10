/*
 * lib.c
 *
 *  Created on: Nov 7, 2014
 *      Author: dev
 */


/*
 * lib.c
 *
 *  Created on: Oct 24, 2014
 *      Author: dev
 */

#include <stdio.h>
#include <stdlib.h>
#include <appMacros.h>
#include <string.h>
#include <lib.h>

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
