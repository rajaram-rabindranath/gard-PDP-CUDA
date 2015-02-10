#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <appMacros.h>
#include <unistd.h>
#include <stdint.h>
#include <lib.h>
#include <string.h>


#define BLOCK_COUNT 5

uint32_t cong_seeded ( uint32_t *jcong );
double cpu_time ( );
uint32_t kiss_seeded ( uint32_t *jcong, uint32_t *jsr, uint32_t *w, uint32_t *z );
uint32_t mwc_seeded ( uint32_t *w, uint32_t *z );
float r4_exp ( uint32_t *jsr, uint32_t ke[256], float fe[256], float we[256] );
void r4_exp_setup ( uint32_t ke[256], float fe[256], float we[256] );
float r4_nor ( uint32_t *jsr, uint32_t kn[128], float fn[128], float wn[128] );
void r4_nor_setup ( uint32_t kn[128], float fn[128], float wn[128] );
float r4_uni ( uint32_t *jsr );
uint32_t shr3_seeded ( uint32_t *jsr );
void timestamp ( );


void printBucket(float* bucket,int low, int high);
void printUsage(char* appName);
int parseArgs(char** argv,int* pSize,int *print,int argc);
float partition_for_K(float data[],int p,int r);
void  swap(float *a,float *b);
float kthsmallest(float  data[],int n, int k);
uint64_t getTime();


void random_number_generator_normal(float* arr, int size, int max_number);


/**
	EACH BLOCK SHALL HAVE a THREAD TO
	SORT THAT BLOCK 
**/
__global__ void cuSort(float* data,int bucketSize,int* startPoint)
{

//	int L= blockIdx.x * blockDim.x;
	int L= blockIdx.x*bucketSize;
	int U= L + bucketSize;
	int j;
	float tmp;
	startPoint[blockIdx.x] = L;
	for(int i=L+1; i < U; i++)
	{
		tmp=data[i];
		j = i-1;
		while(tmp<data[j] && j>=0)
		{
			data[j+1] = data[j];
			j = j-1;
		}
		data[j+1]=tmp;
	}
}


int main(int argc, char** argv)
{
	// check for argument count
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

	// we already know the number of blocks --- FIND BUCKET SIZE
	bucketSize=problemSize/BLOCK_COUNT;
	if(print==TRUE)
	{
		printf("size of each bucket %d\n",bucketSize);
	}

	// mem to get the random numbers from the rand gen
	float* dev_mem=NULL;
	int* dev_dbg = NULL;		
	float* input=(float*) calloc(problemSize,sizeof(float));
	cudaMalloc((void**)&dev_mem, problemSize * sizeof (float));
	cudaMalloc((void**)&dev_dbg, problemSize * sizeof (int));

	/**
	 * 1. Get data from the distribution
	 * 2. Split the data into buckets
	 * 3. Sort the data in each bucket
	 * 4. Append all the post sorted data -- and we done
	 **/
	random_number_generator_normal(input,problemSize,problemSize*10);

	#if DEBUG
	printf("============== PRIOR TO SORTING ====================\n");
	for(int i=0;i<problemSize;i++)
	{
	  printf("RNG at index %d we have %f\n",i,input[i]);
	}
	#endif

	float** bucket_ref = (float**) malloc(BLOCK_COUNT*sizeof(float*));
	float results[BLOCK_COUNT];
	for(int i=0;i<BLOCK_COUNT;i++)
	{
		int sm=(i+1)*bucketSize;
		float result=kthsmallest(input,problemSize,sm);
		results[i]=result;
		#if DEBUG
		printf("%d smallest element is %f ",sm,result);
		#endif
	}

	/**
	 * Partition data into buckets
	 */
	for(int i=0;i<BLOCK_COUNT;i++)
	{

		//printf("===========from %d to %d======\n",i*bucketSize,bucketSize+(i*bucketSize));
		bucket_ref[i] = (float*)malloc(sizeof(float)*bucketSize);
		float result=results[i];
		int max=problemSize*10+1;
		for(int j=i*bucketSize,k=0; k<bucketSize&&j<bucketSize+i*bucketSize;j++,k++)
		{
			if((input[j] <= result))
			{
				bucket_ref[i][k]=input[j];
				input[j]=max;
			}
		}
	}

	/**
	 * Print DATA in buckets
	 */
	#if DEBUG
	for(int i=0;i<BLOCK_COUNT;i++)
	{
		printf("The bucket ====== %d\n",i);
		for(int j=0;j<bucketSize;j++)
		{
			printf("Bucket %d value %f\n",i,bucket_ref[i][j]);
		}
	}
	#endif

	// re-arrange all the buckets data in original memory "input"
	for(int i=0;i<BLOCK_COUNT;i++)
	{
		memcpy((void*)(input+(i*bucketSize)),(void*)bucket_ref[i],bucketSize*sizeof(float));
	}

	/*for(int i=0;i<problemSize;i++)
	{
		printf("The values are %f\n",input[i]);		
	}*/

	cudaMemcpy(dev_mem,input,problemSize * sizeof(float),cudaMemcpyHostToDevice);
	double start, end;
	start = getTime();
	cuSort<<<BLOCK_COUNT,1>>>(dev_mem,bucketSize,dev_dbg);
	end = getTime();
	cudaMemcpy(input,dev_mem, problemSize * sizeof(float),cudaMemcpyDeviceToHost );

	int* dbg=(int *) calloc(BLOCK_COUNT,sizeof(int));
	cudaMemcpy(dbg,dev_dbg, BLOCK_COUNT * sizeof(int),cudaMemcpyDeviceToHost );
	printf("time taken to sort %d numbers is %f seconds. \n",problemSize,(end-start)/1000000.0);
	#if DEBUG
	for(int i=0;i<BLOCK_COUNT;i++)
	{
		printf("Block %d start index %d\n",i,dbg[i]);				
	}
	#endif
	if(print==TRUE)
	{
		printf("============== POST SORTING ====================\n");
		for(int i=0;i<problemSize;i++)
		{
			printf("%f\n",input[i]);
		}
	}


	float a=input[0];
	for(int i=1;i<problemSize;i++)
	{
		if(a > input[i])
		{
			printf("WE HAVE A MAJOR PROBLEM %f\n",input[i]);
			break;
		}
		a=input[i];
				
	}
	
	cudaFree(dev_mem);
	return SUCCESS;
}


void printBucket(float* bucket,int low, int high)
{
	for(int i=low;i<high;i++)
	{
		printf("the val %f \n",bucket[i]);
	}

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


void  swap(float *a,float *b)
{
	float t=*a;
	*a=*b;
	*b=t;
}

float kthsmallest(float data[],int size, int k)
{
	int i=0,foundK=0;
	int p,q;
	float r;
	p=0;
	r=size-1;
	k--;
	while(!foundK)
	{
		q=partition_for_K(data,p,r);
		if(q==k)
		{
			foundK=1;
		}
		else if(k<q)
		{
			r=q-1;
		}
		else
		{
			p=q+1;
		}
	}
	return data[k];
}


float partition_for_K(float data[],int p,int r)
{
	int i,j;
	float pivot;
	i=p-1;
	pivot=data[r];
	for(j=p;j<r;j++)
	{
	   if(data[j]<pivot)
	  {
		i++;
		swap(&data[j],&data[i]);
	   }
	}
	swap(&data[i+1],&data[r]);
	return i+1;
}

uint64_t getTime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (uint64_t)(t.tv_sec)*1000000 + (uint64_t)(t.tv_usec);
}


void random_number_generator_normal(float* arr, int size, int max_number)
{
	uint32_t kn[128];
	float fn[128], wn[128];
	r4_nor_setup ( kn, fn, wn);
	float rnd;
	uint32_t seed = (uint32_t)time(NULL);
	float var = sqrt(max_number);
	for ( int i = 0; i < size; i++)
	{
		rnd = r4_nor(&seed, kn, fn, wn);
		arr[i] = max_number/2 + rnd*var;
	}
}






/******************************************************************************/

uint32_t cong_seeded ( uint32_t *jcong )

/******************************************************************************/
/*
  Purpose:

    CONG_SEEDED evaluates the CONG congruential random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    16 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JCONG, the seed, which is updated 
    on each call.

    Output, uint32_t CONG_SEEDED, the new value.
*/
{
  uint32_t value;

  *jcong = 69069 * ( *jcong ) + 1234567;

  value = *jcong;

  return value;
}
/******************************************************************************/

double cpu_time ( )

/******************************************************************************/
/*
  Purpose:

    CPU_TIME returns the current reading on the CPU clock.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 December 2008

  Author:

    John Burkardt

  Parameters:

    Output, double CPU_TIME, the current reading of the CPU clock, in seconds.
*/
{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}
/******************************************************************************/

uint32_t kiss_seeded ( uint32_t *jcong, uint32_t *jsr, uint32_t *w, uint32_t *z )

/******************************************************************************/
/*
  Purpose:

    KISS_SEEDED evaluates the KISS random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    15 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JCONG, uint32_t *JSR, uint32_t *W, uint32_t *Z, 
    the seeds, which are updated on each call.

    Output, uint32_t KISS_SEEDED, the new value.
*/
{
  uint32_t value;

  value = ( mwc_seeded ( w, z ) ^ cong_seeded ( jcong ) ) + shr3_seeded ( jsr );

  return value;
}
/******************************************************************************/

uint32_t mwc_seeded ( uint32_t *w, uint32_t *z )

/******************************************************************************/
/*
  Purpose:

    MWC_SEEDED evaluates the MWC multiply-with-carry random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    15 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *W, uint32_t *Z, the seeds, which are updated 
    on each call.

    Output, uint32_t MWC_SEEDED, the new value.
*/
{
  uint32_t value;

  *z = 36969 * ( *z & 65535 ) + ( *z >> 16 );
  *w = 18000 * ( *w & 65535 ) + ( *w >> 16 );

  value = ( *z << 16 ) + *w;

  return value;
}
/******************************************************************************/

float r4_exp ( uint32_t *jsr, uint32_t ke[256], float fe[256], float we[256] )

/******************************************************************************/
/*
  Purpose:

    R4_EXP returns an exponentially distributed single precision real value.

  Discussion:

    The underlying algorithm is the ziggurat method.

    Before the first call to this function, the user must call R4_EXP_SETUP
    to determine the values of KE, FE and WE.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    15 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JSR, the seed.

    Input, uint32_t KE[256], data computed by R4_EXP_SETUP.

    Input, float FE[256], WE[256], data computed by R4_EXP_SETUP.

    Output, float R4_EXP, an exponentially distributed random value.
*/
{
  uint32_t iz;
  uint32_t jz;
  float value;
  float x;

  jz = shr3_seeded ( jsr );
  iz = ( jz & 255 );

  if ( jz < ke[iz] )
  {
    value = ( float ) ( jz ) * we[iz];
  }
  else
  {
    for ( ; ; )
    {
      if ( iz == 0 )
      {
        value = 7.69711 - log ( r4_uni ( jsr ) );
        break;
      }

      x = ( float ) ( jz ) * we[iz];

      if ( fe[iz] + r4_uni ( jsr ) * ( fe[iz-1] - fe[iz] ) < exp ( - x ) )
      {
        value = x;
        break;
      }

      jz = shr3_seeded ( jsr );
      iz = ( jz & 255 );

      if ( jz < ke[iz] )
      {
        value = ( float ) ( jz ) * we[iz];
        break;
      }
    }
  }
  return value;
}
/******************************************************************************/

void r4_exp_setup ( uint32_t ke[256], float fe[256], float we[256] )

/******************************************************************************/
/*
  Purpose:

    R4_EXP_SETUP sets data needed by R4_EXP.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    14 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Output, uint32_t KE[256], data needed by R4_EXP.

    Output, float FE[256], WE[256], data needed by R4_EXP.
*/
{
  double de = 7.697117470131487;
  int i;
  const double m2 = 2147483648.0;
  double q;
  double te = 7.697117470131487;
  const double ve = 3.949659822581572E-03;

  q = ve / exp ( - de );

  ke[0] = ( uint32_t ) ( ( de / q ) * m2 );
  ke[1] = 0;

  we[0] = ( float ) ( q / m2 );
  we[255] = ( float ) ( de / m2 );

  fe[0] = 1.0;
  fe[255] = ( float ) ( exp ( - de ) );

  for ( i = 254; 1 <= i; i-- )
  {
    de = - log ( ve / de + exp ( - de ) );
    ke[i+1] = ( uint32_t ) ( ( de / te ) * m2 );
    te = de;
    fe[i] = ( float ) ( exp ( - de ) );
    we[i] = ( float ) ( de / m2 );
  }
  return;
}
/******************************************************************************/

float r4_nor ( uint32_t *jsr, uint32_t kn[128], float fn[128], float wn[128] )

/******************************************************************************/
/*
  Purpose:

    R4_NOR returns a normally distributed single precision real value.

  Discussion:

    The value returned is generated from a distribution with mean 0 and 
    variance 1.

    The underlying algorithm is the ziggurat method.

    Before the first call to this function, the user must call R4_NOR_SETUP
    to determine the values of KN, FN and WN.

    Thanks to Chad Wagner, 21 July 2014, for noticing a bug of the form
      if ( x * x <= y * y );   <-- Stray semicolon!
      {
        break;
      }

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 July 2014

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JSR, the seed.

    Input, uint32_t KN[128], data computed by R4_NOR_SETUP.

    Input, float FN[128], WN[128], data computed by R4_NOR_SETUP.

    Output, float R4_NOR, a normally distributed random value.
*/
{
  int hz;
  uint32_t iz;
  const float r = 3.442620;
  float value;
  float x;
  float y;

  hz = ( int ) shr3_seeded ( jsr );
  iz = ( hz & 127 );

  if ( fabs ( hz ) < kn[iz] )
  {
    value = ( float ) ( hz ) * wn[iz];
  }
  else
  {
    for ( ; ; )
    {
      if ( iz == 0 )
      {
        for ( ; ; )
        {
          x = - 0.2904764 * log ( r4_uni ( jsr ) );
          y = - log ( r4_uni ( jsr ) );
          if ( x * x <= y + y )
          {
            break;
          }
        }

        if ( hz <= 0 )
        {
          value = - r - x;
        }
        else
        {
          value = + r + x;
        }
        break;
      }

      x = ( float ) ( hz ) * wn[iz];

      if ( fn[iz] + r4_uni ( jsr ) * ( fn[iz-1] - fn[iz] ) 
        < exp ( - 0.5 * x * x ) )
      {
        value = x;
        break;
      }

      hz = ( int ) shr3_seeded ( jsr );
      iz = ( hz & 127 );

      if ( fabs ( hz ) < kn[iz] )
      {
        value = ( float ) ( hz ) * wn[iz];
        break;
      }
    }
  }

  return value;
}
/******************************************************************************/

void r4_nor_setup ( uint32_t kn[128], float fn[128], float wn[128] )

/******************************************************************************/
/*
  Purpose:

    R4_NOR_SETUP sets data needed by R4_NOR.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    14 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Output, uint32_t KN[128], data needed by R4_NOR.

    Output, float FN[128], WN[128], data needed by R4_NOR.
*/
{
  double dn = 3.442619855899;
  int i;
  const double m1 = 2147483648.0;
  double q;
  double tn = 3.442619855899;
  const double vn = 9.91256303526217E-03;

  q = vn / exp ( - 0.5 * dn * dn );

  kn[0] = ( uint32_t ) ( ( dn / q ) * m1 );
  kn[1] = 0;

  wn[0] = ( float ) ( q / m1 );
  wn[127] = ( float ) ( dn / m1 );

  fn[0] = 1.0;
  fn[127] = ( float ) ( exp ( - 0.5 * dn * dn ) );

  for ( i = 126; 1 <= i; i-- )
  {
    dn = sqrt ( - 2.0 * log ( vn / dn + exp ( - 0.5 * dn * dn ) ) );
    kn[i+1] = ( uint32_t ) ( ( dn / tn ) * m1 );
    tn = dn;
    fn[i] = ( float ) ( exp ( - 0.5 * dn * dn ) );
    wn[i] = ( float ) ( dn / m1 );
  }

  return;
}
/******************************************************************************/

float r4_uni ( uint32_t *jsr )

/******************************************************************************/
/*
  Purpose:

    R4_UNI returns a uniformly distributed real value.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JSR, the seed.

    Output, float R4_UNI, a uniformly distributed random value in
    the range [0,1].
*/
{
  uint32_t jsr_input;
  float value;

  jsr_input = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = fmod ( 0.5 
    + ( float ) ( jsr_input + *jsr ) / 65536.0 / 65536.0, 1.0 );

  return value;
}
/******************************************************************************/

uint32_t shr3_seeded ( uint32_t *jsr )

/******************************************************************************/
/*
  Purpose:

    SHR3_SEEDED evaluates the SHR3 generator for integers.

  Discussion:

    Thanks to Dirk Eddelbuettel for pointing out that this code needed to
    use the uint32_t data type in order to execute properly in 64 bit mode,
    03 October 2013.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 October 2013

  Author:

    John Burkardt

  Reference:

    George Marsaglia, Wai Wan Tsang,
    The Ziggurat Method for Generating Random Variables,
    Journal of Statistical Software,
    Volume 5, Number 8, October 2000, seven pages.

  Parameters:

    Input/output, uint32_t *JSR, the seed, which is updated 
    on each call.

    Output, uint32_t SHR3_SEEDED, the new value.
*/
{
  uint32_t value;

  value = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = value + *jsr;

  return value;
}
/******************************************************************************/

void timestamp ( )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
