import java.io.*;
import java.util.*;

public class GFG {
	public static void main (String[] args) {
	    Scanner sc = new Scanner(System.in);
	    int t = sc.nextInt();
	    while(t-- > 0){
	        int n = sc.nextInt();
	        int k = sc.nextInt();
	        int[] a = new int[k];
	        int[] temp = new int[k];
	        for(int i=0;i<k;i++){
	            a[i] = sc.nextInt();
	        }
	        for(int i=0;i<k;i++){
	            temp[i] = sc.nextInt();
	        }
	        long[] c = new long[n];
	        Arrays.fill(c,Integer.MAX_VALUE);
	        for(int i=0;i<k;i++){
	            c[a[i]-1] = temp[i];
	        }
	        long p = Integer.MAX_VALUE;
	        long[] left = new long[n];
	        for(int i=0;i<n;i++){
	            p = (p+1<c[i])?p+1:c[i];
	            left[i] = p;
	        }
	        p = Integer.MAX_VALUE;
	        long[] right = new long[n];
	        for(int i=n-1;i>=0;i--){
	            p = (p+1<c[i])?p+1:c[i];
	            right[i] = p;
	        }
	        for(int i=0;i<n;i++){
	            long kl = (left[i]>right[i])?right[i]:left[i];
	            System.out.print(kl+" ");
	        }
	        System.out.println();
	    }
	}
}
