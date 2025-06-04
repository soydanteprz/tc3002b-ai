import java.io.*;
import java.util.*;

public class Main {

	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		PrintWriter out = new PrintWriter(System.out);

		/******  CODE STARTS HERE  *****/
		//--------------------------------------------------------------------------------------------------------
		int t = fs.nextInt();
		while(t-->0) {
			int n = fs.nextInt();
			int[] h = fs.readArray(n);
			int l=1, r=(int)1e9;
			int ans = -1;
			while(l<=r) {
				int mid = (l+r)/2;
				if(isPossible(mid, h, n)) {
					ans = mid;
					l = mid+1;
				}else {
					r= mid-1;
				}
			}
			out.println(ans);
		}
		out.close();
	}
	
	static boolean isPossible(int min, int[] h, int n) {
		int[] c = new int[n];
		for(int i=n-1; i>=2; i--) {
			
			if(h[i]+c[i]-min<0)return false;
			
			int x  = Math.min(h[i], h[i]+c[i]-min) / 3;
			c[i-1] += x;
			c[i-2] += 2*x;
		}
		
		return (h[0]+c[0]>=min&&h[1]+c[1]>=min);
	}
    
	//******  CODE ENDS HERE  *****
	//----------------------------------------------------------------------------------------------------------------
	
	
	
	
	//----------- FastScanner class for faster input---------------------------
	
}
