import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.*;


 
public class Codeforces {
	
	static int mod=1000000007 ;
	static List<Integer>[] adj;
	static boolean vst[];
	static int dp[];
	public static void main(String[] args) throws Exception {
		PrintWriter out=new PrintWriter(System.out);
		FastScanner fs=new FastScanner();
		int t=fs.nextInt();
		while(t-->0) {
			int n=fs.nextInt();
			int arr[]=fs.readArray(n);
			int ans[]=new int[n];
			if(n%2==0) {
				for(int i=0;i<n;i+=2) {
					ans[i]=-arr[i+1];
					ans[i+1]=arr[i];
				}
			}
			else {
				for(int i=3;i<n;i+=2) {
					ans[i]=-arr[i+1];
					ans[i+1]=arr[i];
				}
				int a=0, b=0, c=0;
				outer:for(int i=0;i<3;i++) {
					for(int j=i+1;j<3;j++) {
						if(arr[i]+arr[j]!=0) {
							b=i;
							c=j;
							a= 3-c-b;
							break outer;
						}
					}
				}
				ans[a]=arr[b]+arr[c];
				ans[b]=-arr[a];
				ans[c]=-arr[a];
			}
			for(int i=0;i<n;i++) {
				out.print(ans[i]+" ");
			}
			out.println();
//			long sum=0;
//			for(int i=0;i<n;i++) {
//				sum+=arr[i]*ans[i];
//			}
//			if(sum!=0) System.out.println(false);
		}
		out.close();
		
	}
	
	
	
	
	
	
	
	
	
	
	// Use this to input code since it is faster than a Scanner
	
 
}
