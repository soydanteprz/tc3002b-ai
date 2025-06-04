import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
public class a{ 
    public static void main(String args[]) throws java.lang.Exception{ 
    	FastScanner s=new FastScanner();
		PrintWriter out=new PrintWriter(System.out);
		int t=s.nextInt();
		for(int tt=0;tt<t;tt++) {
			int n=s.nextInt(),k=s.nextInt();
			int pos[]=s.readArray(k);
			int temp[]=s.readArray(k);
			long ans[]=new long[n];
			Arrays.fill(ans,Integer.MAX_VALUE);
			for(int i=0;i<k;i++){
				ans[pos[i]-1]=temp[i];
			}
			for(int i=1;i<n;i++){
				ans[i]=Math.min(ans[i-1]+1,ans[i]);
			}
			for(int i=n-2;i>=0;i--){
				ans[i]=Math.min(ans[i],ans[i+1]+1);
			}
			for(int i=0;i<n;i++){
				out.print(ans[i]+" ");
			}
			out.println();
		}
		out.close();
    }
	
    
	
    
    
    
    
    
    
}
