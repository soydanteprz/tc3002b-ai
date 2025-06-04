import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Stack;
import java.util.StringTokenizer;

public class D {

	public static void main(String[] args) throws NumberFormatException, IOException {
		BufferedReader file = new BufferedReader(new InputStreamReader(System.in));
		int n = Integer.parseInt(file.readLine());
		int[] heights = new int[n];
		StringTokenizer st = new StringTokenizer(file.readLine());
		for(int i = 0; i < n; i++) {
			heights[i] = Integer.parseInt(st.nextToken());
		}
		int[] dp = new int[n];
		dp[0] = 0;
		Stack<Integer> high = new Stack<>();
		Stack<Integer> low = new Stack<>();
		high.push(0);
		low.push(0);
		for(int i = 1; i < n; i++) {
			dp[i] = dp[i-1]+1;
			while(!high.isEmpty() && heights[high.peek()] < heights[i]) {
				dp[i] = Math.min(dp[i], dp[high.peek()]+1);
				high.pop();
			}
			if(!high.isEmpty()) {
				dp[i] = Math.min(dp[i], dp[high.peek()]+1);
				if(heights[high.peek()] == heights[i])
					high.pop();
			}
			while(!low.isEmpty() && heights[low.peek()] > heights[i]) {
				dp[i] = Math.min(dp[i], dp[low.peek()]+1);
				low.pop();
			}
			if(!low.isEmpty()) {
				dp[i] = Math.min(dp[i], dp[low.peek()]+1);
				if(heights[low.peek()] == heights[i])
					low.pop();
			}
			high.push(i);
			low.push(i);
		}
		System.out.println(dp[n-1]);
	}

}
