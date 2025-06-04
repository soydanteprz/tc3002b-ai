import java.io.*;
import java.util.*;

public class test3 {

	public static void main(String[] args) throws  IOException {
		FastReader f = new FastReader();
		int t = f.nextInt();
		while(t-->0) {
			 int n = f.nextInt();
			 int C[] = new int[n];
			 long ans=Long.MAX_VALUE,pre = 0;
			 PriorityQueue<Integer> epq = new PriorityQueue<Integer>();
			 PriorityQueue<Integer> opq = new PriorityQueue<Integer>();
			 for(int i = 0;i<n;i++) {
				 C[i] = f.nextInt();
				 if(i%2==0)epq.add(C[i]);
				 else opq.add(C[i]);
				 pre+=C[i];
				 if(i>0) {
					 ans  = Math.min(ans,pre+ ((long)n-(long)epq.size())*(long)epq.peek()
							 		+ ((long)n-(long)opq.size())*(long)opq.peek());
				 }
			 }
			 System.out.println(ans);
			}
    }
	
	
	
	

	
	
	
    
	  
		
	 
	 

 }
