import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

public class D {
	
	public static void main(String[] args) throws IOException {
		FastScanner fs = new FastScanner();
		PrintWriter out = new PrintWriter(System.out);
		
		int t = fs.nextInt();
		for (int tc = 0; tc < t; tc++) {
			int n = fs.nextInt();
			int[] a = fs.readArray(n);
			String s = fs.nextLine();
			
			// let all blue to be 1 -> blueCount
			ArrayList<Integer> blues = new ArrayList<Integer>();
			ArrayList<Integer> reds = new ArrayList<Integer>();
			
			for (int i = 0; i < s.length(); i++) {
				if (s.charAt(i) == 'B') {
					blues.add(a[i]);
				} else {
					reds.add(a[i]);
				}
			}
			
			Collections.sort(blues);
			Collections.sort(reds);
			
			boolean ok = true;
			
			for (int i = 1; i <= blues.size(); i++) {
				if (blues.get(i - 1) < i) {
					ok = false;
					break;
				}
			}
			
			for (int i = blues.size() + 1; i <= n; i++) {
				if (reds.get(i - blues.size() - 1) > i) {
					ok = false;
					break;
				}
			}
			
			if (ok) {
				out.println("YES");
			} else {
				out.println("NO");
			}
		}
		
		out.close();
	}
	
	
}
