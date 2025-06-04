import java.io.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.*;
import java.io.*;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.*;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.math.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Vector;
import static java.lang.Math.sqrt;
import static java.lang.Math.floor;






public class topcoder {
   
	


	
	
	
static class pairr implements  Comparable<pairr>{
	Integer value;
	Integer index;
	
	   public pairr(Integer value, Integer index) {
		    this.value = value;
		    this.index = index;
	   }
	   
}



	 
	
	
	 
	 
	 
	 
	 
	 
	 
		     
		     
		     
		     
			  
			  
			  
			      
			    // Prints all prime numbers smaller than 'n'
			
			    
			    
			    
			 
               
            
			  public static int add(long n, long num, long a, int i) {
				//System.out.println(num);
				  if(num > n)return -1;
				
				  if(num == n) {
					  
					  return i;
				  }
				  
				if(a < 2050)return -1;
				  
				
				 
				  long temp = num+a;
				  
				  if(temp <= n) {
				return	  add(n,temp,a,i+1);
				  }else if(temp > n){
					  a /= 10;
					  return add(n,num,a,i);
				  }
				  
			      return -1;
				  }
			
				   
		
		
		
		
		
		
		
		
		
		  
		  

		    
		    
		    
		    
		    
		
		
			
	    
	   
	  
			
	    public static void main(String args[])throws IOException{
			 
	    	//	 System.setIn(new FileInputStream("Case.txt"));
	    		 BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
	    	    
	    		 
	    		 int t = Integer.parseInt(ob.readLine());
	    		 
	    		 
	    		 while( t--> 0) {
	    		
	    		StringTokenizer	 st = new StringTokenizer(ob.readLine());
	    			 int n = Integer.parseInt(st.nextToken());
	    			 int m = Integer.parseInt(st.nextToken());
	    			 int x = Integer.parseInt(st.nextToken());
	    			 
	    			 int []ar = new int[n];
	    			 st = new StringTokenizer(ob.readLine());
	    			 PriorityQueue<Integer>pq = new PriorityQueue<>();
	    			 
	    			 for(int i = 0; i < n; i++) {
	    				 ar[i] = Integer.parseInt(st.nextToken());
	    				 pq.add(ar[i]);
	    			 }
	    			 
	    			 
	    	PriorityQueue<pairr>p = new PriorityQueue<>();
	    	for(int i =0; i < n; i++) {
	    		p.add(new pairr(ar[i],i));
	    	}
	    	int []ans = new int[n];
	    	
	    	
	    	for(int i = 0; i < n; i++) {
	    	pairr a = p.poll();
	    	ans[a.index] = (i%m)+1;
	    	}
	    	
	    	System.out.println("YES");
	    	for(int i = 0; i < n; i++ ) {
	    		System.out.print(ans[i]+" ");
	    	}
	    	System.out.println();
	    		 }
	    }
}
