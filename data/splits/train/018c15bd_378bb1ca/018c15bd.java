import java.util.*;
import java.util.Map.Entry;


import java.io.*;
import java.math.*;
import static java.util.stream.Collectors.*;
import static java.util.Map.Entry.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Hey
{
	public static void main (String[] args) throws IOException
	{
		final long mod=(long) (1e9+7);  
		
		Reader s=new Reader();
		  PrintWriter pt=new PrintWriter(System.out);
//		  BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		  int T=s.nextInt();
//		  int T=Integer.parseInt(br.readLine());
//		  int T=1;
		  while(T-->0)
		  {
			  int n=s.nextInt();
			  int c[]=new int[n];
			  for(int i=0;i<n;i++) {
				  c[i]=s.nextInt();
			  }
			  long mo[]=new long[n];
			  long me[]=new long[n];
			  mo[0]=-1;
			  me[0]=-1;
			  mo[1]=c[0];
			  me[1]=c[1];
			  long so[]=new long[n];
			  long se[]=new long[n];
			  so[1]=c[0];
			  se[1]=c[1];
			  for(int i=2;i<n;i++) {
				  if(i%2==0) {
					  mo[i]=Math.min(c[i], mo[i-1]);
					  me[i]=me[i-1];
					  so[i]=so[i-1]+c[i];
					  se[i]=se[i-1];
				  }
				  else {
					  mo[i]=mo[i-1];
					  me[i]=Math.min(c[i], me[i-1]);
					  so[i]=so[i-1];
					  se[i]=se[i-1]+c[i];
				  }
			  }
			  
//			  pa(mo);
//			  pa(me);
//			  pa(so);
//			  pa(se);
			  long min=Long.MAX_VALUE;
			  for(int i=1;i<n;i++) {
				  long k1=0, k2=0;
				  if(i%2==1) {
					  k1=i/2+1;
					  k2=i/2+1;
				  }
				  else {
					  k2=i/2;
					  k1=k2+1;
				  }
				  min=Math.min((n-k1)*mo[i]+so[i]+(n-k2)*me[i]+se[i], min);
			  }
			  pt.println(min);
		  }
		  pt.close();
	}
	
	
	
	

	

	

	

	
	
	
	
	
	
	

	

	

			 //S%N, N is a power of 2

	
	
	
	
	 
	
	
	 
	 




	

	  
	  
	 
	
	
	  

	
	 
	
	
	
	     
	
	
	
	  

	 
     
     
	



	 
	
	
	
	
	 
  
    // Main function that sorts arr[l..r] using 
    // merge() 
     
	 
  
    // Main function that sorts arr[l..r] using 
    // merge() 
     
    
	 
}
