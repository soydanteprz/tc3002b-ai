import java.util.Scanner;

public class C1499 {
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		int t = in.nextInt();
		while (t-- > 0) {
			int n = in.nextInt();
			// int[] arr = new int[n];
			long[] mn = { Long.MAX_VALUE, Long.MAX_VALUE };
			long[] rem = { n, n };
			long sum = 0;
			long ans = Long.MAX_VALUE;
			for (int i = 0; i < n; i++) {
				int temp = in.nextInt();
				mn[i % 2] = Math.min(mn[i % 2], temp);
				rem[i % 2]--;
				sum += temp;
				if (i > 0) {
					long cur = sum + rem[0] * mn[0] + rem[1] * mn[1];
					ans = Math.min(ans, cur);
				}
			}
			System.out.println(ans);

//			int a = Integer.MAX_VALUE;
//			int aIndex = -1;
//			int b = Integer.MAX_VALUE;
//			int bIndex = -1;
//
//			for (int i = 0; i < n; i++) {
//				arr[i] = in.nextInt();
//				if (i % 2 == 0) {
//					if (arr[i] < a) {
//						a = arr[i];
//						aIndex = i;
//					}
//				} else {
//					if (arr[i] < b) {
//						b = arr[i];
//						bIndex = i;
//					}
//				}
//			}
//			int sum = 0;
//			for (int i = 0; i < Math.max(bIndex, aIndex) + 1; i++) {
//				if (i % 2 == 0) {
//					if (i == aIndex) {
//						if (aIndex < bIndex) {
//							sum += (n - (i / 2) - ((bIndex - aIndex) / 2)) * arr[i];
//						} else {
//							sum += (n - (i / 2)) * arr[i];
//						}
//					} else {
//						sum += arr[i];
//					}
//				} else {
//					if (i == bIndex) {
//						if (bIndex < aIndex) {
//							sum += (n - (i / 2) - ((aIndex - bIndex) / 2)) * arr[i];
//						} else {
//							sum += (n - (i / 2)) * arr[i];
//						}
//					} else {
//						sum += arr[i];
//					}
//				}
//
//			}
//			System.out.println(sum);
//			for (int i = 0; i < n; i++) {
//				arr[i] = in.nextInt();
//				if (arr[i] < a) {
//					a = arr[i];
//					aIndex = i;
//
//				}
//
//			}
//			if (aIndex == 0) {
//
//				for (int i = 1; i < n; i++) {
//					if (arr[i] < b) {
//						b = arr[i];
//						bIndex = i;
//					}
//				}
//				System.out.println(aIndex + " " + bIndex);
//				System.out.println(a + " " + b);
//				int sum = 0;
//				for (int i = 1; i < bIndex; i++) {
//					sum += arr[i];
//				}
//				sum += b * (n - bIndex + 1);
//				sum += a * n;
//				System.out.println(sum);
//			} else {
//				int b = Integer.MAX_VALUE;
//				int bIndex = -1;
//				for (int i = 0; i < aIndex; i++) {
//					if (arr[i] < b) {
//						b = arr[i];
//						bIndex = i;
//					}
//				}
//				System.out.println(aIndex + " " + bIndex);
//				System.out.println(a + " " + b);
//				int sum = 0;
//				for (int i = 0; i < bIndex; i++) {
//					sum += arr[i];
//				}
//				sum += b * (n - bIndex);
//				sum += a * n;
//				System.out.println(sum);
//			}

		}
	}
}
