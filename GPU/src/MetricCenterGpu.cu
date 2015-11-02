//****************************
//File:MetricCenterGpu.cu
//author:Salil Rajadhyaksha
//version:1-Nov-2015
//***************************

//number of thread per block
#define NT 1024

//Structue for 2-D points
typedef struct
{
double x;
double y;
}vector_t;

//Structure for result to store radius and index.
typedef struct
{
double radius;
int index;
}ResultTuple;

//per thread variable in shared memory to store the max dist found by each thread
__shared__ ResultTuple points[NT];

/**
 * Calculate the euclidean distance between two points.
 *
 * @param  a :Index of first point.
 * @param  b  :Index of second point.
 * @param   xy :array that holds the points.
 * @return the euclidean distance.
 */
__device__ double calculateEuclideanDistance(int a,int b,vector_t *xy)
{
	vector_t *pointA=&xy[a];
	vector_t *pointB=&xy[b];
	double dx=pointA->x - pointB->x;
	double dy=pointA->y - pointB->y;
	return sqrt(dx*dx+dy*dy);	
}
/*
find maximum of the two points by comparing their radius
@param a: first point;
@param b: secon point;
@return a:The max of a and b stored in a;
*/
__device__ ResultTuple *findMaxDistance( ResultTuple *a ,ResultTuple *b)
{
	if(a->radius < b->radius)	
		a->radius=b->radius;
		
	return a;
}

/*
Return  the minimum of the two points.
@param :a first point;
@param :b second point;

@returns :a:the minimum stored in a;
*/
__device__ ResultTuple *reduce(ResultTuple *a, ResultTuple *b)
{
	if(a->radius>b->radius||a->radius==-1)
	{
		a->radius=b->radius;
		a->index=b->index;
	}
	
	return a;
}

/**

return maximum between two doubles

@param x: pointer to first double value;
@param y: second value
@return max stored in x;
*/
__device__ double *returnMax(double *x,double y)
{
	if(*x<y)
	*x=y;	 
	return x;
 }
/**
Device Kernel to find the smallest distance between a set of points per block.
Called with a one dimensional grid.
Blocks= number of multiprocessors.
thread=1024 per block.
Each block calculates max distance for one point at a time in a for scheduled in leapfrog.
The threads calculate the max distance for that one point in the inner for schdeuled again in leapfrog.

@param :xy-List of points 
@param :N- total number of points.
@param :result- the array to store the semi-final result per block;
*/
extern "C" __global__ void calculateRadius(vector_t *xy,int N,ResultTuple* result)
{
int thr,bID,numberOfBlocks;
double max;
thr=threadIdx.x;
bID=blockIdx.x;
numberOfBlocks=gridDim.x;

for(unsigned long long int i=bID;i<=N;i+=numberOfBlocks)//schedule points to blocks in leapfrog pattern.
	{						  
		max=0.00;	
		for(unsigned long long int j=thr;j<N;j+=NT)//calculate distance of for current point with respect to all points in leapfrog.
			{
				if(j==i)
					continue;	
				returnMax(&max,calculateEuclideanDistance(i,j,xy));	//call to find the max passing the indices of points and current max.		
			}
		points[thr]=(ResultTuple){max,i};//storing the max in the shared memory.
		
__syncthreads();
//calculate the maximum for that point via shared memory parallel reduction.
   for (int k =NT/2;k>0;k>>=1)
      {
      if (thr<k)
         {
			findMaxDistance(&points[thr],&points[thr+k]);
         }
__syncthreads();
      }
	  //single threaded section.
	if(thr==0)
	{
		reduce(&result[bID],&points[thr]);//reduce to store if current point is less than the min so far for this block.
	}

	}
	
}