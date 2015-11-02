import java.io.File;
import java.io.FileNotFoundException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Scanner;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;

/*
 * 
 * Class:MetricCenterGpu.java
 * @author:Salil Rajadhyaksha
 * @version:01-Nov-2015
 * 
 * Class that reads the points from the file and initializes the GPU kernel and call the kernel function.
 */
public class MetricCenterGpu extends Task {

	int NT=1024;//number of threads per block
	
	/*
	 * class ResultTuple.java
	 * This class mirrors the ResultTuple struct on the gpu.
	 * THe results from each multiprocessor block of the gpu are stored in this object per block..
	 * @author:Salil Rajadhyaksha
	 * @version:1-Nov-2015.
	 */
	private static class ResultTuple extends Struct
	{

		public double radius;
		public int index;
		
		/*
		 * Constructor
		 * @param index: index of point.
		 * @param radius:radius of point.
		 */
		public ResultTuple(int index,double radius)
		{
			this.radius=radius;
			this.index=index;
			
		}
		//returns size in bytes of size of struct in C
		public static long sizeof()
		{
			return 16;
		}
		
		/*
		 * Find the set of points with the minimum points and print result.
		 * @param: result-The points returned from the gpu after block level reduction.
		 * @param: vectors- The list of points.
		 */
		public static void reduceAndPrintResult(GpuStructArray<ResultTuple>result,GpuStructArray<Vector>vectors)
		{
			ResultTuple min=result.item[0];
			for(int i=1;i<result.length();i++)
			{
				if(result.item[i].radius<min.radius)
				{
					min=result.item[i];
				}
			}
			System.out.print(min.index+" "+"(");
			System.out.printf ("%.5g",vectors.item[min.index].x);
			System.out.print(",");
			System.out.printf ("%.5g", vectors.item[min.index].y);
			System.out.println(")");
			System.out.printf ("%.5g",min.radius);
			System.out.println();
			
		}
		/*Read this java object from the given byte buffer as a C Struct
		 * 
		 * @param :byte buffer.
		 */
		@Override
		public void fromStruct(ByteBuffer buf) {

			radius=buf.getDouble();
			index=buf.getInt();
		}

		/*
		 * Write this java object to the given byte buffer as a C struct.
		 * @param:buf-byte buffer.
		 */
		@Override
		public void toStruct(ByteBuffer buf) {
			
			buf.putDouble(radius);
			buf.putInt(index);
		}		
		
	}
	/*
	 * class Vector.java
	 * This class mirrors the vector_t struct on the gpu.
	 * THe points taken from the file are stored in objects of this class.
	 * @author:Salil Rajadhyaksha
	 * @version:1-Nov-2015.
	 */
	private static class Vector extends Struct
	{

		public double x;
		public double y;
		
		/*
		 * Constructor
		 * @param -x :first point
		 * @param -y :second point
		 */
		public Vector(double x,double y)
		{
			this.x=x;
			this.y=y;
		}
		
		//returns size in bytes of size of struct in C
		public static long sizeof()
		{
			return 16;
		}
		
		/*Read this java object from the given byte buffer as a C Struct
		 * 
		 * @param :byte buffer.
		 */
		@Override
		public void fromStruct(ByteBuffer buf) {

			x=buf.getDouble();
			y=buf.getDouble();
		}

		/*
		 * Write this java object to the given byte buffer as a C struct.
		 * @param:buf-byte buffer.
		 */
		@Override
		public void toStruct(ByteBuffer buf) {

			buf.putDouble(x);
			buf.putDouble(y);
		}
			
	}
	/*
	 * Kernel function interface.
	 * @author:Salil Rajadhyaksha
	 * @version:1-nov-2015
	 */
	private static interface DistanceKernal extends Kernel
	{
		public void calculateRadius(GpuStructArray<Vector> vectors,int n,GpuStructArray<ResultTuple> result);
	}
	
	/*
	 *Main program of the task.Intializes the kernel grid.
	 *Read the points from the file.puts the points on the gpu
	 *call the the kernel function on the GPU.
	 *makes call to evalute the final result from the semi-final result.
	 *
	 *@params:args[]-command line arguments.
	 */
	@Override
	public void main(String[] args) throws Exception {

		if(args.length!=1)//if number of arguments incorrect.
		{
			System.out.println("Invalid number of arguments");
			return;
		}
		Scanner read=null;
		
		try {
			read=new Scanner(new File(args[0]));
		} catch (FileNotFoundException e) {
			System.out.println("File could not be read.");
		}
		
				
		ArrayList<Double>xPoints=new ArrayList<>();
		ArrayList<Double>yPoints=new ArrayList<>();
		
		
		//read all points and put it in ArrayList.
		while(read.hasNext())
		{
			xPoints.add(read.nextDouble());
			yPoints.add(read.nextDouble());
		}
		read.close();

		int count=xPoints.size();//number of points.
		
		if(count<2)
		{
			System.out.println("Invalid number of points");
		}
		
		//set up gpu variables
		Gpu gpu=Gpu.gpu();
		gpu.ensureComputeCapability(2,0);
		Module module=gpu.getModule("MetricCenterGpu.cubin");
		
		GpuStructArray<Vector>vectors;
		vectors=gpu.getStructArray(Vector.class, count);
		
		GpuStructArray<ResultTuple> result=gpu.getStructArray(ResultTuple.class, gpu.getMultiprocessorCount());
		
		for(int i=0;i<result.length();i++)
			result.item[i]=new ResultTuple(0,-1);
		
		for(int i=0;i<count;i++)
		{
			vectors.item[i]=new Vector(xPoints.get(i),yPoints.get(i));
		}
		vectors.hostToDev();
		result.hostToDev();
		//set up kernal
		DistanceKernal kernal=module.getKernel(DistanceKernal.class);
		kernal.setBlockDim(NT);
		kernal.setGridDim(gpu.getMultiprocessorCount());
		kernal.calculateRadius(vectors,count,result);//call to kernal function.
	
		result.devToHost();	
		
		ResultTuple.reduceAndPrintResult(result,vectors);//call to reduce and print result
			
	}

	/*
	 * specify that this task requires 1 Cpu core.
	 */
   protected static int coresRequired()
      {
      return 1;
      }

   /**
    * Specify that this task requires one GPU accelerator.
    */
   protected static int gpusRequired()
      {
      return 1;
      }

}
