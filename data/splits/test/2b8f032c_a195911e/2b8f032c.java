import java.io.*;
import java.util.*;

public class Main {

	static int INF = (int)1e9 + 1;
	
	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		PrintWriter out = new PrintWriter(System.out);

		/******  CODE STARTS HERE  *****/
		//--------------------------------------------------------------------------------------------------------
		int t = fs.nextInt();
		w:while(t-->0) {
			int n = fs.nextInt();
			int[] k = fs.readArray(n);
			int[] h = fs.readArray(n);
			long ans = 0;
			int mtn = INF, prev = k[n-1];
			
			for(int j=n-1; j>=0; j--) {
				if(mtn!=INF && mtn > k[j]) {
					int x = prev-mtn+1;
					ans += ((long)x*(x+1))/2;
					mtn = INF;
					prev = k[j];
				}
				if(mtn >= k[j]-h[j]+1) 
					mtn = k[j]-h[j]+1;
				
				if(j==0) {
					int x = prev-mtn+1;
					ans += ((long)x*(x+1))/2;
				}
			}
			out.println(ans);
		}
		out.close();
	}
    
	//******  CODE ENDS HERE  *****
	//----------------------------------------------------------------------------------------------------------------
	
	
	
	
	//----------- FastScanner class for faster input---------------------------
	
}
