import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Menorah {
	
	
	
	public static void solve(int n, String a, String b) {
		
		int bothOne = 0;
		int bothZero = 0;
		int oneZero = 0;
		int zeroOne = 0;
		for (int i=0; i<n; i++) {
			if (a.charAt(i) == '0') {
				if (b.charAt(i) == '0') {
					bothZero++;
				}
				else {
					zeroOne++;
				}
			}
			else {
				if (b.charAt(i) == '0') {
					oneZero++;
				}
				else {
					bothOne++;
				}
			}
		}
		int ans = Integer.MAX_VALUE;
		if (bothOne - bothZero == 1) {
			ans = (bothZero*2) + 1;
		}
		if (oneZero == zeroOne) {
			ans = Math.min(ans, oneZero*2);
		}
		if (ans == Integer.MAX_VALUE) {
			ans = -1;
		}
		System.out.println(ans);
		
	}

	public static void main(String[] args) {
		
		FastReader in = new FastReader();
		int t = in.nextInt();
		while (t-- > 0) {
			solve(in.nextInt(), in.next(), in.next());
		}

	}

}
