import java.util.*;

public class Main {
	static Scanner scan = new Scanner(System.in);

	static int[] readArray(int[] x) {
		for(int i=0; i<x.length; ++i) x[i] = scan.nextInt();
		return x;
	}

	static long[] readArray(long[] x) {
		for(int i=0; i<x.length; ++i) x[i] = scan.nextLong();
		return x;
	}


	

	public static void main(String[] args) {
		int t = scan.nextInt();
		for(int it=0; it<t; ++it) {
			int n = scan.nextInt();
			long[] aa = readArray(new long[n]);

			long minEven = aa[0];
			long minOdd = aa[1];
			long sum = aa[0]+aa[1];
			long best = n*minEven + n*minOdd;
			int numOdd = 1;
			int numEven = 1;
			for(int i=2; i<n; ++i) {
				if(i%2 == 0) {
					minEven = Math.min(aa[i], minEven);
					numEven++;
				}else {
					minOdd = Math.min(aa[i], minOdd);
					numOdd++;
				}
				sum += aa[i];
				long score = sum;
				score += minEven*(n-numEven);
				score += minOdd*(n-numOdd);				
				best = Math.min(best, score);
			}
			System.out.println(best);
		}
	}
}
