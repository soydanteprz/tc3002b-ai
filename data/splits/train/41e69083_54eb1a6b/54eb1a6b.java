import java.util.*;
public class Solution {
 
	public static void main(String[] args) {
		// TODO Auto-generated method stub
          Scanner io =new Scanner (System.in);
          int t = io.nextInt();
  		for (int iii=0; iii<t; iii++) {
  			int n = io.nextInt();
  			long k = io.nextLong();
   
  			Long[] arr = new Long[n];
  			for (int i=0; i<n; i++) {
  				arr[i] = io.nextLong();
  			}
   
  			Arrays.sort(arr);
  			long[] psum = new long[n];
  			psum[0] = arr[0];
  			for (int i=1; i<n; i++) {
  				psum[i] = psum[i-1] + arr[i];
  			}
   
  			if (psum[n-1] <= k) {
  				System.out.println(0);
  				continue;
  			}
  			//System.out.println(psum[n-1]);
   
  			long ans = Long.MAX_VALUE;
  			
  			for (int i=0; i<n; i++) {
  				long sum = psum[i] + (n-i-1)*arr[0];
  				if (sum <= k) {
  					ans = Math.min(ans, n-1-i);
  					continue;
  				}
  				long ops = (long) Math.ceil((double) (sum - k)/(double) (n-i));
   
  				ans = Math.min(ans, n-1-i + ops);
  				//System.out.println(sum);
  			}
  			System.out.println(ans);
  			
  		}
 
        	  }
        	   
	
	
 
 
}
