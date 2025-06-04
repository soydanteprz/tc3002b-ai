import java.io.*;
import java.util.*;

public class Menorah {
	public static void main(String[] args) throws Exception {
		FastIO in = new FastIO();
		int t = in.nextInt();
		for (int tc=0; tc<t; tc++) {
			int n = in.nextInt();
			String original = in.next();
			String target = in.next();
			int numDiff = 0;
			int original1 = 0;
			int target1 = 0;
			for (int i=0; i<n; i++) {
				if (original.charAt(i)!=target.charAt(i)) {
					numDiff++;
				}
				if (original.charAt(i)=='1') original1++;
				if (target.charAt(i)=='1') target1++;
			}
			int evenAns = Integer.MAX_VALUE;
			int oddAns = Integer.MAX_VALUE;
			if (original1==target1) {
				evenAns = numDiff;
			}
			for (int i=0; i<n; i++) {
				if (original.charAt(i)=='1' && target.charAt(i)=='1') {
					int ones = (n-original1)+1;
					if (ones==target1) {
						oddAns = n-numDiff;
					}
					break;
				}
			}
			for (int i=0; i<n; i++) {
				if (original.charAt(i)=='1' && target.charAt(i)=='0') {
					int ones = (n-original1)+1;
					if (ones==target1) {
						oddAns = Math.min(n-numDiff+1, oddAns);
					}
					break;
				}
			}
			int ans = Math.min(evenAns, oddAns);
			if (ans==Integer.MAX_VALUE) System.out.println(-1);
			else System.out.println(ans);
		}
	}
	
}
