import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.util.*;

public class B {
	static int tree[];
	static int up[];
	static int n;
	static int mod =1000000007; 
	public static void main(String[] args) throws Exception {
		PrintWriter out=new PrintWriter(System.out);
	    FastScanner fs=new FastScanner();
	    int t=fs.nextInt();
	   outer: while(t-->0) {
		    n=fs.nextInt();
		    int k=fs.nextInt();
		   int ac[]=fs.readArray(k);
		   int temp[]=fs.readArray(k);
		   int arr[]=new int[n];
		   Arrays.fill(arr, Integer.MAX_VALUE/2);
		   for(int i=0;i<k;i++) {
			   int ind = ac[i]-1;
			   arr[ind]=temp[i];
		   }
		   int ans[]=new int[n];
		   int left[]=new int[n];
		   Arrays.fill(left, Integer.MAX_VALUE/2);
//		   int right[]=new int[n];
		   left[0]=arr[0];
//		   Arrays.fill(right, Integer.MAX_VALUE/2);
		   
//		   right[n-1]=arr[n-1];
		   for(int i=1;i<n;i++) {
			   left[i]=Math.min(arr[i], left[i-1]+1);
		   }
		   ans[n-1]=Math.min(left[n-1],arr[n-1]);
		   for(int i=n-2;i>=0;i--) {
			   ans[i]=Math.min(Math.min(arr[i],left[i]),ans[i+1]+1);
		   }
//		   for(int i=0;i<n;i++) ans[i]=Math.min(left[i], right[i]);
		   for(int i=0;i<n;i++) out.print(ans[i]+" ");
		   out.println();
	   }
	    out.close();
	}
	
	
	
	
	
	
	
	
	
	// Use this to input code since it is faster than a Scanner
	
 
}
