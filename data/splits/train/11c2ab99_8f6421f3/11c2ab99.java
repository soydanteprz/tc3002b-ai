import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.StringTokenizer;

public class Main {

	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		int T=fs.nextInt();
		PrintWriter out=new PrintWriter(System.out);
		for (int tt=0; tt<T; tt++) {
			int n=fs.nextInt(), k=fs.nextInt();
			int[] positions=fs.readArray(k), temps=fs.readArray(k);
			int[] forced=new int[n];
			Arrays.fill(forced, Integer.MAX_VALUE/2);
			for (int i=0; i<k; i++) forced[positions[i]-1]=temps[i];
			for (int i=1; i<n; i++)
				forced[i]=Math.min(forced[i], forced[i-1]+1);
			for (int i=n-2; i>=0; i--)
				forced[i]=Math.min(forced[i], forced[i+1]+1);
			for (int i=0; i<n; i++) out.print(forced[i]+" ");
			out.println();
			
		}
		out.close();
	}

	
	
	

	
}
