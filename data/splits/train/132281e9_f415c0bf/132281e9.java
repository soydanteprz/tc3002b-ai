import java.io.*;
import java.math.*;
import java.util.*;


public class test {
	 
	   
	   
	 
	   
	   
	    
	 
	   
	
	static boolean function(long a[],long x) {
		
		long arr[] = new long[a.length];
		
		for(int i=0;i<arr.length;i++) {
			arr[i] = a[i];
		}
		
		for(int i=arr.length-1;i>=2;i--) {
			
			if(arr[i] < x) {
				return false;
			}
			long d = Math.min(a[i],(arr[i]-x))/3;
			
			arr[i-1] += d;
			arr[i-2] += 2*d;
		}
		
		return (arr[0]>=x && arr[1]>=x);
	}
	   public static void main(String args[]) throws Exception { 
		
		   Scanner sc = new Scanner();
		   StringBuilder res = new StringBuilder();
		
		   int tc = sc.nextInt();
		  
		   while(tc-->0) { 
			  
			int n = sc.nextInt();
			
			long a[] = new long[n];
			
			for(int i=0;i<n;i++) {
				a[i] = sc.nextLong();
			}
			
			long l = 1;
			long r = 1000000000;
			
			long ans = 1;
			
			while(l<=r){
				
				long mid = l + (r-l)/2;
				
				if(function(a,mid)){
					ans = mid;
					l = mid+1;
				}
				else {
					r = mid-1;
				}
			}
			
			res.append(ans+"\n");
		   }
		   System.out.println(res);
	   }
}
