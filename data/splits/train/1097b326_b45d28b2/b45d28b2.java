import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class MinimumGridPath {
	
	public static void main(String[] args) throws IOException {
		FastIO in = new FastIO(System.in);
		
		int t = in.nextInt();
		while (t-- > 0) {
			int n = in.nextInt();
			long[] c = new long[n];
			long[] sums = new long[n];
			long[] mins = new long[n];
			
			for (int i = 0; i < n; i++) c[i] = in.nextInt();
			
			sums[0] = mins[0] = c[0];
			sums[1] = mins[1] = c[1];
			for (int i = 2; i < n; i++) {
				sums[i] = sums[i-2] + c[i];
				mins[i] = Math.min(mins[i-2], c[i]);
			}
			
			long ans = Long.MAX_VALUE;
			for (int i = 1; i < n; i++) {
				ans = Math.min(ans, (n - (long) Math.ceil((double) (i+1) / 2)) * mins[i] + (n - (i+1) / 2) * mins[i-1] + sums[i] + sums[i-1]);
			}
			
			System.out.println(ans);
		}
		
		in.close();
	}
	
	
}
