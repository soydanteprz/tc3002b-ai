import java.util.*;
import java.io.*;
import java.math.*;
import java.lang.*;

public class MinimumGridPath {
    
    static int mod = 1000000007;

    

    public static void main(String[] args) throws Exception {
        Reader scn = new Reader();
        PrintWriter pw = new PrintWriter(System.out);
        int t = scn.nextInt();
        outer : while(t-->0){
            int n = scn.nextInt();
            long[] arr = new long[n];
            for(int i=0; i<n; i++){
            	arr[i] = scn.nextInt();
            }
            long ans = Long.MAX_VALUE;
           	int k = 2;
           	long oddSum = arr[0];
           	long evenSum = 0;
           	long oddMin = arr[0];
           	long evenMin = Long.MAX_VALUE;
           	long oddCount = 1;
           	long evenCount = 0;
           	while(k <= n){
           		if(k % 2 == 1){
           			oddSum += arr[k-1];
           			oddCount++;
           			oddMin = Math.min(oddMin, arr[k-1]);
           		}else{
           			evenSum += arr[k-1];
           			evenCount++;
           			evenMin = Math.min(evenMin, arr[k-1]);
           		}
           		long sum = oddSum - oddMin + oddMin*(n - oddCount + 1) + evenSum - evenMin + evenMin*(n - evenCount + 1);
           		ans = Math.min(ans, sum);
           		k++;
           	}
           	pw.println(ans);
        }
        pw.close();
    }

    

    

    

    

    

    

    // (x^y)%p in O(logy)
    

    
    
    
    
    

    // TC- O(logmax(a,b))
    

    

    

    // TC- O(logmax(a,b))
    

    

     
}
