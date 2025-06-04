import java.util.*;
import java.io.*;

public class BalancedStones {

	public static void main(String[] args) throws IOException {
		BufferedReader f = new BufferedReader(new InputStreamReader(System.in));
		int t = Integer.parseInt(f.readLine());
		while (t > 0) {
			t--;
			int n = Integer.parseInt(f.readLine());
			int[] arr = new int[n];
			StringTokenizer st = new StringTokenizer(f.readLine());
			for (int i = 0; i < n; i++) {
				arr[i] = Integer.parseInt(st.nextToken());
			}
			int low = 0;
			int high = 1000000000;
			int answer = -1;
			while (low <= high) {
				int mid = low + (high - low) / 2;
				if (check(mid, arr)) {
					answer = mid;
					low = mid + 1;
				} else {
					high = mid - 1;
				}
			}
			System.out.println(answer);
		}
	}

	public static boolean check(int value, int[] old) {
		int[] newarr = new int[old.length];
		for (int i = 0; i < old.length; i++) {
			newarr[i] = old[i];
		}
		for (int i = old.length - 1; i >= 0; i--) {
			if (newarr[i] < value)
				return false;
			if (i > 1) {
				int d = Math.min(newarr[i] - value, old[i]) / 3;
				newarr[i] -= 3 * d;
				newarr[i - 1] += d;
				newarr[i - 2] += 2 * d;
			}
		}
		return true;
	}
}
