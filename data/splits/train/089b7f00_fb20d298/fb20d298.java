import java.io.*;
import java.util.*;
public class Main {
	public static void main(String[] args) {
		FastScanner sc = new FastScanner();
		PrintWriter out = new PrintWriter(System.out);
//		int t=1;	
		int t=sc.nextInt();
//		outer:	
		while(t-- >0) {
			int n = sc.nextInt();
			int b[] = sc.readArray(n);
			TreeSet<Integer> set = new TreeSet<>();
			boolean f = true;
			set.add(b[0]);
			for(int i=1 ; i<n ; i++) {
				set.add(b[i]);
				if(b[i-1]==b[i]) {
					continue;
				}
				Integer find = set.lower(b[i]);
				if(find!=null && find==b[i-1]) {
					continue;
				}
				find = set.higher(b[i]);
				if(find!=null && find==b[i-1]) {
					continue;
				}
				f = false;
				break;
			}
			if(f == false) {
				out.println("NO");
			}
			else out.println("YES");
		}
		out.flush();
		out.close();
	}
	
}
