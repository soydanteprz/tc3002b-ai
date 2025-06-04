import java.util.*;

public class CF763C {
	private static final int MAX = 1000000001;
	private static final int MIN = 0;

	public static final void main(String ...args) {
		Scanner sc = new Scanner(System.in);

		int t = sc.nextInt();

		while (t-- > 0) {
			int n = sc.nextInt();
			int[] h = new int[n];
			int i, b = MAX, e = MIN;
			for (i = 0; i < n; i++) {
				h[i] = sc.nextInt();
				if (b > h[i])
					b = h[i];
				if (e < h[i])
					e = h[i];
			}

			int[] s = new int[n];
			int c, d, ans = b;

			// System.out.println(b + " " + e);

			while (b <= e) {
				c = (b + e) / 2;
				// System.out.println(c);
				// sc.next();
				for (i = 0; i < n; i++) {
					s[i] = 0;
				}

				for (i = n - 1; i >= 2; i--) {
					if (h[i] + s[i] < c) {
						e = c - 1;
						break;
					} else {
						d = Math.min(h[i], h[i] + s[i] - c) / 3;
						s[i - 1] += d;
						s[i - 2] += 2 * d;
					}
				}
				// System.out.println(i);
				if (i == 1) {
					if (h[i] + s[i] < c || h[i - 1] + s[i - 1] < c) {
						e = c - 1;
					} else {
						ans = c;
						b = c + 1;	
					}
				}
			}
			System.out.println(ans);
		}
	}
}
