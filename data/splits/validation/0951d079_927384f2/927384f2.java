import java.util.*;
import java.io.*;

public class Main{
	      
	
	public static void main(String[] args) throws IOException{
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		int n = pint(in);
		Stack<Integer> min = new Stack<Integer>();
		Stack<Integer> max = new Stack<Integer>();
		
		int[] a = new int[n];
		int[] dp = new int[n];
		StringTokenizer st = new StringTokenizer(in.readLine());
		a[0] = pint(st);
		min.add(0);
		max.add(0);
		for(int i = 1; i < n; i++) {
			a[i] = pint(st);
			int h = dp[i - 1] + 1;
			while(!max.isEmpty() && a[i] > a[max.peek()]) {
				int k = a[max.peek()];
				h = Math.min(h, dp[max.pop()] + 1);
				while(!max.isEmpty() && a[max.peek()] == k) {max.pop();}
			}
			if(!max.isEmpty()) {
				h = Math.min(h, dp[max.peek()] + 1);
			}
			while(!min.isEmpty() && a[i] < a[min.peek()]) {
				int k = a[min.peek()];
				h = Math.min(h, dp[min.pop()] + 1);
				while(!min.isEmpty() && a[min.peek()] == k) {min.pop();}
			}
			if(!min.isEmpty()) {
				h = Math.min(h, dp[min.peek()] + 1);
			}
			dp[i] = h;
			min.add(i);
			max.add(i);
		}
		System.out.println(dp[n - 1]);
	}
	
    static int pint(BufferedReader in) throws IOException {return Integer.parseInt(in.readLine());}
    static int pint(StringTokenizer st) {return Integer.parseInt(st.nextToken());}
}
