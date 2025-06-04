import java.util.*;
import java.io.*;

public class D_1525 {
	
	static int INF = (int)1e9;
	static int n, m;
	static int[] full, free;
	static int[][] memo;
	
	public static int dp(int i, int j) {
		if(i == n)
			return 0;
		if(j == m)
			return INF;
		
		if(memo[i][j] != -1)
			return memo[i][j];
		
		return memo[i][j] = Math.min(dp(i, j + 1), Math.abs(free[j] - full[i]) + dp(i + 1, j + 1));
	}
	
	public static void main(String[] args) throws Exception {
		Scanner sc = new Scanner(System.in);
		PrintWriter pw = new PrintWriter(System.out);
		
		int l = sc.nextInt();
		int[] array = sc.nextIntArray(l);
		
		n = 0;
		for(int i = 0; i < l; i++)
			if(array[i] == 1)
				n++;
		m = l - n;
		
		full = new int[n];
		free = new int[m];
		int ind1 = 0, ind2 = 0;
		for(int i = 0; i < l; i++)
			if(array[i] == 0)
				free[ind2++] = i;
			else
				full[ind1++] = i;
		
		memo = new int[n][m];
		for(int[] i : memo)
			Arrays.fill(i, -1);
		
		pw.println(dp(0, 0));
		pw.flush();
	}

	
}
