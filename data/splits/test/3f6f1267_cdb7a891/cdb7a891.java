import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;

import java.lang.*;

public class Practice {
	public static long mod = 998244353;
	public static long mod2 = 998244353;
	public static long tt = 0;
	public static int[] ttt = new int[2];
	public static long[] fac = new long[500000];
	public static ArrayList<Integer> prime;

	public static void main(String[] args) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		PrintWriter pw = new PrintWriter(System.out);

		// int t = Integer.parseInt(br.readLine());
		// while (t-- > 0) {
		fac();
		String[] s1 = br.readLine().split(" ");
		int n = Integer.parseInt(s1[0]);
		int k = Integer.parseInt(s1[1]);
		Long[][] arr = new Long[n][2];
		Long[] st = new Long[n];
		Long[] end = new Long[n];
		for (int i = 0; i < n; i++) {
			String str = (br.readLine());
			String[] s2 = str.split(" ");
			arr[i][0] = Long.parseLong(s2[0]);
			st[i] = arr[i][0];
			arr[i][1] = Long.parseLong(s2[1]);
			end[i] = arr[i][1];
		}
		Arrays.sort(st);
		Arrays.sort(end);
		long pp = (power(fac[k], mod - 2));
		long[] comb = new long[n + 1];
		for (int i = 1; i <= n; i++) {
			if (i < k) {
				continue;
			}
			// System.out.println(i + " " + fac[i] + " " + power(fac[k], mod - 2) + " " +
			// fac[k]);
			comb[i] = (fac[i] * pp) % mod;
			comb[i] = (comb[i] * (power(fac[i - k], mod - 2))) % mod;
		}
//		for (int i = 0; i <= n; i++) {
//			System.out.print(comb[i] + " ");
//		}
		HashMap<Long, Integer> map1 = new HashMap<Long, Integer>();
		HashMap<Long, Integer> map = new HashMap<>();
		for (int i = 0; i < n; i++) {
			if (map.containsKey(arr[i][0])) {
				map.put(arr[i][0], map.getOrDefault(arr[i][0], 0) + 1);
				continue;
			}
			int a = getLower(arr[i][0] - 1, st);
			int b = getLower(arr[i][0] - 1, end);
			map.put(arr[i][0], map.getOrDefault(arr[i][0], 0) + 1);
			map1.put(arr[i][0], map1.getOrDefault(arr[i][0], 0));
			if (a != -1 || b != -1) {
				if (b == -1) {
					int curr = a + 1;
					map1.put(arr[i][0], map1.getOrDefault(arr[i][0], 0) + curr);
					continue;
				}
				// System.out.println(arr[i][0] + " " + b + " " + a);
				int curr = a - b;
				map1.put(arr[i][0], map1.getOrDefault(arr[i][0], 0) + curr);
			}
		}
		long ans = 0;
		for (long a : map.keySet()) {
			ans = (ans + comb[map.get(a) + map1.get(a)] + mod - comb[map1.get(a)]) % mod;
		}
		// System.out.println(map + " " + map1);
		pw.println(ans);
		// }
		pw.close();

	}

	private static long power(long a, long p) {
		// TODO Auto-generated method stub
		long res = 1;
		while (p > 0) {
			if (p % 2 == 1) {
				res = (res * a) % mod;
			}
			p = p / 2;
			a = (a * a) % mod;
		}
		return res;
	}

	private static void fac() {
		fac[0] = 1;
		// TODO Auto-generated method stub
		for (int i = 1; i < fac.length; i++) {
			if (i == 1) {
				fac[i] = 1;
			} else {
				fac[i] = i * fac[i - 1];
			}
			if (fac[i] > mod) {
				fac[i] = fac[i] % mod;
			}
		}
	}

	private static int getLower(Long long1, Long[] st) {
		// TODO Auto-generated method stub
		int left = 0, right = st.length - 1;
		int ans = -1;
		while (left <= right) {
			int mid = (left + right) / 2;
			if (st[mid] <= long1) {
				ans = mid;
				left = mid + 1;
			} else {
				right = mid - 1;
			}
		}
		return ans;
	}

}
