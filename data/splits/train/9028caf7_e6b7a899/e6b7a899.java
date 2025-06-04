import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class Main {
	static final long MOD=1000000007;
	static final long MOD1=998244353;
	static long ans=0;
	//static ArrayList<Integer> ans=new ArrayList<>();
	public static void main(String[] args){
		PrintWriter out = new PrintWriter(System.out);
		InputReader sc=new InputReader(System.in);
		int N = sc.nextInt();
		int[] A = sc.nextIntArray(N);
		ArrayList<Integer> a1 = new ArrayList<Integer>();
		ArrayList<Integer> a2 = new ArrayList<Integer>();
		for (int i = 0; i < A.length; i++) {
			if (A[i]==0) {
				a1.add(i);
			}else {
				a2.add(i);
			}
		}
		int[][] dp = new int[a1.size()+1][a2.size()+1];
		for (int i = 0; i < dp.length; i++) {
			Arrays.fill(dp[i], Integer.MAX_VALUE/2);
		}
		dp[0][0] = 0;
		for (int i = 1; i <= a1.size() ; i++) {
			int pos1 = a1.get(i-1);
			for (int j = 0; j <= a2.size(); j++) {
				dp[i][j] = dp[i-1][j];
				if (j-1>=0) {
					int pos2 = a2.get(j-1);
					dp[i][j] = Math.min(dp[i][j], dp[i-1][j-1] + Math.abs(pos1-pos2));
				}
			}
		}
		System.out.println(dp[a1.size()][a2.size()]);
	}
	
}
