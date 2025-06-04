import java.io.*;
import java.util.*;

public class Main {
	
	
	public static void main(String[] args) throws Exception {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out));		
		
		int T = Integer.parseInt(in.readLine());
		
		for (int ts=1; ts<=T; ts++) {
			int N = Integer.parseInt(in.readLine());
			String A = in.readLine();
			String B = in.readLine();
			
			int ans = -1;
			
			//even number of ops
			int fromZero = 0;
			int fromOne = 0;
			for (int i=0; i<N; i++) {
				if (A.charAt(i) != B.charAt(i)) {
					if (A.charAt(i) == '0') {
						fromZero++;
					} else {
						fromOne++;
					}
				}
			}
			if (fromZero == fromOne) {
				ans = fromZero + fromOne;
			}
			
			
			//odd number of ops
			int zero = 0;
			int one = 0;
			for (int i=0; i<N; i++) {
				if (A.charAt(i) == B.charAt(i)) {
					if (A.charAt(i) == '0') {
						zero++;
					} else {
						one++;
					}
				}
			}
			
			if (one == zero + 1) {
				int current = one + zero;
				if (ans == -1)
					ans = current;
				else
					ans = Math.min(ans, current);
			}
			
			out.write(ans + "\n");
		}
		
		in.close();
		out.close();
	}

}
