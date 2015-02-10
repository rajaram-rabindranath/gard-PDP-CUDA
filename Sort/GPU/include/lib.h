/*
 * lib.h
 *
 *  Created on: Oct 24, 2014
 *      Author: dev
 */

#ifndef LIB_H_
#define LIB_H_

#include <stdint.h>

void printBucket(float* bucket,int low, int high);
void printUsage(char* appName);
int parseArgs(char** argv,int* pSize,int *print,int argc);
float partition_for_K(float A[],int p,int r);
void  swap(float *a,float *b);
float kthsmallest(float  A[],int n, int k);
uint64_t getTime();

#endif /* LIB_H_ */
