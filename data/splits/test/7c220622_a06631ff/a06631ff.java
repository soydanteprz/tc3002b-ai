import java.io.*;
import java.util.*;

public class Main {
	static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	static StringTokenizer st;
	static PrintWriter pr = new PrintWriter(new OutputStreamWriter(System.out));
	static String readLine() throws IOException {
		return br.readLine();
	}
	static String next() throws IOException {
		while (st == null || !st.hasMoreTokens())
			st = new StringTokenizer(readLine());
		return st.nextToken();
	}
	static int readInt() throws IOException {
		return Integer.parseInt(next());
	}
	
	
	
	
	static void solve() throws IOException {
		int n = readInt(), a[] = new int[n];
		for (int i = 0; i < n; ++i) a[i] = readInt();
		for (int i = 0; i < n; ++i) {
			List<Integer> b = new ArrayList();
			for (int j = 0; j < n; ++j)
				if (j != i) b.add(a[j]);
			int s = (int)Math.pow(3, n - 1);
			for (int mm = 0; mm < s; ++mm) {
				int sum = 0;
				for (int m = mm, j = 0; m > 0; m /= 3, ++j) {
					if (m%3 == 1) sum += b.get(j);
					else if (m%3 == 2) sum -= b.get(j);
				}
				if (sum == a[i]) {
					pr.println("YES");
					return;
				}
			}
		}
		pr.println("NO");
	}
	public static void main(String[] args) throws IOException {
		//solve();
		for (int t = readInt(); t > 0; --t) solve();
		pr.close();
	}
}
