import java.util.*;
import java.io.*;

public class E_1547 {
	public static void main(String[] args) throws Exception {
		Scanner sc = new Scanner(System.in);
		PrintWriter pw = new PrintWriter(System.out);
		
		int T = sc.nextInt();
		while(T-->0) {
			int n = sc.nextInt(), k = sc.nextInt();
			int[] a = sc.nextIntArray(k);
			int[] t = sc.nextIntArray(k);
			
			int[] array = new int[n];
			Arrays.fill(array, Integer.MAX_VALUE);
			for(int i = 0; i < k; i++)
				array[a[i] - 1] = t[i];
			
			int[] pre = new int[n];
			int[] post = new int[n];
			
			int prev = (int)2e9;
			for(int i = 0; i < n; i++)
				prev = pre[i] = Math.min(prev + 1, array[i]);
			
			prev = (int)2e9;
			for(int i = n - 1; i >= 0; i--)
				prev = post[i] = Math.min(prev + 1, array[i]);
			
			for(int i = 0; i < n; i++)
				array[i] = Math.min(pre[i], post[i]);
			
			for(int i = 0; i < n; i++)
				pw.print(array[i] + (i == n - 1 ? "\n" : " "));
		}
		
		pw.flush();
	}

	
}
