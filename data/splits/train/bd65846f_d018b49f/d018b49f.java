import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Codechef {
	static long dp[][];
	public static void main(String[] args) throws java.lang.Exception {
		FastReader in = new FastReader(System.in);
		StringBuilder sb = new StringBuilder();
		int t = 1;
		//t = in.nextInt();
		while (t > 0) {
			--t;
			int n = in.nextInt();
			int arr[] = new int[n];
			List<Integer> ones = new ArrayList<Integer>();
			List<Integer> zero = new ArrayList<>();
			for(int i = 0;i<n;i++)
			{
				int a = in.nextInt();
				if(a == 1)
					ones.add(i);
				else 
					zero.add(i);
			}
			if(ones.size() == 0) {
				sb.append(0+"\n");
				continue;
			}
			dp = new long[ones.size()][zero.size()];
			for(int i = 0;i<ones.size();i++)
			    Arrays.fill(dp[i], -1);
			sb.append(findans(ones, zero, ones.size()-1, zero.size()-1));
           
		}
		System.out.print(sb);
	}
	
	static long findans(List<Integer> ones,List<Integer> zero,int x,int y)
	{
		if(x < 0)
			return 0;
		if(y<0)
			return Integer.MAX_VALUE;
        if(dp[x][y]!=-1)
        	return dp[x][y];
		return  dp[x][y] = Math.min(findans(ones, zero, x, y-1),findans(ones, zero, x-1, y-1) + (long)Math.abs(ones.get(x)-zero.get(y)));
	}

	

	

	
}


