import java.util.*;
import java.lang.*;
import java.io.*;

public class Codechef {
	static long fans[] = new long[200001];
	static long inv[] = new long[200001];
	static long mod = 1000000007;

	

	

	public static void main(String[] args) throws java.lang.Exception {
		FastReader in = new FastReader(System.in);
		StringBuilder sb = new StringBuilder();
		int t = 1;
		t = in.nextInt();
		while (t > 0) {
			--t;
			int n = in.nextInt();
			long time[] = generateArray(in, n);
			long hp[] = generateArray(in, n);
			int s = 0;
			long ans = 0;
			while(s<time.length)
			{
				long l = time[s] - hp[s];
				long r = time[s];
				for(int i = s+1;i<n;i++)
				{
					if(time[i]-hp[i]<=l)
					{
						l = time[i]-hp[i];
						r = time[i];
					}
					else if(time[i]-hp[i]<r)
					{
						r = time[i];
					}
				}
					
					while(s<n && time[s]>=l && time[s]<=r)
					{
						++s;
					}
					long temp = r - l;
					ans += (temp*(temp+1))/2;
			}
			sb.append(ans+"\n");

		}
		System.out.print(sb);
	}

	

	static long[] generateArray(FastReader in, int n) throws IOException {
		long arr[] = new long[n];
		for (int i = 0; i < n; i++)
			arr[i] = in.nextLong();
		return arr;
	}

	

	

	

	
}


