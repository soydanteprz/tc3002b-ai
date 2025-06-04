import java.util.*;
import java.io.*;

public class C_1551 {
	public static void main(String[] args) throws Exception {
		Scanner sc = new Scanner(System.in);
		PrintWriter pw = new PrintWriter(System.out);
		
		int t = sc.nextInt();
		while(t-->0) {
			int n = sc.nextInt();
			int[] a = new int[n];
			int[] b = new int[n];
			int[] c = new int[n];
			int[] d = new int[n];
			int[] e = new int[n];
			for(int i = 0; i < n; i++) {
				String s = sc.next();
				int[] cnt = new int[5];
				for(int j = 0; j < s.length(); j++)
					cnt[s.charAt(j) - 'a']++;
				a[i] = 2 * cnt[0] - s.length();
				b[i] = 2 * cnt[1] - s.length();
				c[i] = 2 * cnt[2] - s.length();
				d[i] = 2 * cnt[3] - s.length();
				e[i] = 2 * cnt[4] - s.length();
			}
			
			Arrays.sort(a);
			Arrays.sort(b);
			Arrays.sort(c);
			Arrays.sort(d);
			Arrays.sort(e);
			
			int max = 0;
			int sum = 0;
			for(int i = n - 1; i >= 0; i--) {
				sum += a[i];
				if(sum <= 0)
					break;
				max = n - i;
			}
			
			sum = 0;
			for(int i = n - 1; i >= 0; i--) {
				sum += b[i];
				if(sum <= 0)
					break;
				max = Math.max(max, n - i);	
			}
			
			sum = 0;
			for(int i = n - 1; i >= 0; i--) {
				sum += c[i];
				if(sum <= 0)
					break;
				max = Math.max(max, n - i);	
			}
			
			sum = 0;
			for(int i = n - 1; i >= 0; i--) {
				sum += d[i];
				if(sum <= 0)
					break;
				max = Math.max(max, n - i);	
			}
			
			sum = 0;
			for(int i = n - 1; i >= 0; i--) {
				sum += e[i];
				if(sum <= 0)
					break;
				max = Math.max(max, n - i);	
			}
			
			pw.println(max);
			
		}
		
		pw.flush();
	}

	
}
