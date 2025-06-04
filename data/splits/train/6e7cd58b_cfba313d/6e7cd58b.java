import java.util.*;
import java.io.*;

public class CodeforcesRound734 {
	static FastReader sc = new FastReader();

	public static void main(String[] args) throws IOException {
		try {
			int t = sc.nextInt();
			while (t-- > 0) {
//				A();
//				B1();
//				B2();
				C();
			}
		} catch (Exception e) {
//			return;
			e.printStackTrace();
		}
	}

	static void C() {
		int n = sc.nextInt();
		int a[][] = new int[5][n];
		for (int i = 0; i < n; i++) {
			String s = sc.next();
			for (int j = 0; j < s.length(); j++) {
				if (s.charAt(j) == 'a')
					a[0][i]++;
				else
					a[0][i]--;
				if (s.charAt(j) == 'b')
					a[1][i]++;
				else
					a[1][i]--;
				if (s.charAt(j) == 'c')
					a[2][i]++;
				else
					a[2][i]--;
				if (s.charAt(j) == 'd')
					a[3][i]++;
				else
					a[3][i]--;
				if (s.charAt(j) == 'e')
					a[4][i]++;
				else
					a[4][i]--;
			}
		}
		for (int x[] : a) {
			Arrays.sort(x);
		}
		int ans = 0;
		for (int i = 0; i < 5; i++) {
			int temp = 0, cnt = 0;
			for (int j = n - 1; j >= 0; j--) {
				temp += a[i][j];
				if (temp <= 0)
					break;
				else
					cnt++;
			}
			ans = Math.max(ans, cnt);
		}
		System.out.println(ans);
	}

	

	

	

	

	

	

}


