import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;
import java.util.StringTokenizer;

// 1 0 1 0 0
public class D {

	// Right side greater
	public static int[] rightGreaterEle(long[] a) {
		int n = a.length;
		int ans[] = new int[n];
		ans[n - 1] = -1;
		Stack<Integer> st = new Stack<>();
		st.add(n - 1);
		for (int i = n - 2; i >= 0; i--) {
			if (st.isEmpty() || a[st.peek()] >= a[i]) {
				ans[i] = st.peek();
				st.push(i);
			} else {

				while (!st.isEmpty() && a[st.peek()] < a[i]) {
					st.pop();
				}
				if (st.isEmpty())
					ans[i] = -1;
				else
					ans[i] = st.peek();
				st.push(i);
			}
		}
		return ans;

	}

	// Right side smaller
	public static int[] rightSideSmaller(long[] a) {
		int n = a.length;
		int ans[] = new int[n];
		ans[n - 1] = -1;
		Stack<Integer> st = new Stack<>();
		st.add(n - 1);
		for (int i = n - 2; i >= 0; i--) {
			if (st.isEmpty() || a[st.peek()] <= a[i]) {
				ans[i] = st.peek();
				st.push(i);
			} else {

				while (!st.isEmpty() && a[st.peek()] > a[i]) {
					st.pop();
				}
				if (st.isEmpty())
					ans[i] = -1;
				else
					ans[i] = st.peek();
				st.push(i);
			}
		}
		return ans;

	}

	// Left side greater
	public static int[] leftSideGreater(long[] a) {
		int n = a.length;
		int ans[] = new int[n];
		ans[0] = -1;
		Stack<Integer> st = new Stack<>();
		st.push(0);
		for (int i = 1; i < n; i++) {
			if (st.isEmpty() || a[st.peek()] >= a[i]) {
				ans[i] = st.peek();
				st.push(i);
			} else {

				while (!st.isEmpty() && a[st.peek()] < a[i]) {
					st.pop();
				}
				if (st.isEmpty())
					ans[i] = -1;
				else
					ans[i] = st.peek();
				st.push(i);
			}
		}
		return ans;

	}

	// Left side smaller
	public static int[] leftSideSmaller(long[] a) {
		int n = a.length;
		int ans[] = new int[n];
		ans[0] = -1;
		Stack<Integer> st = new Stack<>();
		st.push(0);
		for (int i = 1; i < n; i++) {
			if (st.isEmpty() || a[st.peek()] <= a[i]) {
				ans[i] = st.peek();
				st.push(i);
			} else {

				while (!st.isEmpty() && a[st.peek()] > a[i]) {
					st.pop();
				}
				if (st.isEmpty())
					ans[i] = -1;
				else
					ans[i] = st.peek();
				st.push(i);
			}
		}
		return ans;

	}

	public static void main(String[] args) {
		FastScanner sc = new FastScanner();
		int n = sc.nextInt();
		long a[] = new long[n];
		for (int i = 0; i < n; i++)
			a[i] = sc.nextLong();

		int rg[] = rightGreaterEle(a);
		int rs[] = rightSideSmaller(a);

		int lg[] = leftSideGreater(a);
		int ls[] = leftSideSmaller(a);

		int dp[] = new int[n];
		Arrays.fill(dp, Integer.MAX_VALUE);

		dp[0] = 0;
		for(int i=1;i<n;i++) dp[i]=dp[i-1]+1;
		
		for (int i = 0; i < n; i++) {
			
		
			// Where from we can reach at ith index
			//System.out.println(l1+" "+l2+" "+r1+" "+r2);
			int l1 = ls[i], l2 = lg[i];
			if (l1 != -1)
				dp[i] = Math.min(dp[i], dp[l1] + 1);
			if (l2 != -1)
				dp[i] = Math.min(dp[i], dp[l2] + 1);

			//System.out.println("from left "+dp[i]);
			int r1 = rs[i], r2 = rg[i];

			if (r1 != -1)
				dp[r1] = Math.min(dp[r1], dp[i] + 1);
			if (r2 != -1)
				dp[r2] = Math.min(dp[r2], dp[i] + 1);
			
//			if(i==4)
//			System.out.println(dp[l1]+" "+dp[l2]+" "+dp[r1]+" "+dp[r2]);
//			System.out.println((i+1)+" "+dp[i]);

		}
		System.out.println(dp[n-1]);

	}

	

}
