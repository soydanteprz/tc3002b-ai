import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import java.util.*;
import java.io.*;
import java.math.*;

/**
 *
 * @Har_Har_Mahadev
 */

/**
 * Main , Solution , Remove Public
 */
public class A {

	public static void process() throws IOException {

		int n = sc.nextInt();
		String s = sc.next(),t = sc.next();
		int zs = 0, zt = 0;
		for(int i = 0; i<n; i++) {
			if(s.charAt(i) == '0')zs++;
			if(t.charAt(i) == '0')zt++;
		}
		
		int ans = n+1;
		if(zs == zt) {
			int min = 0;
			for(int i = 0; i<n; i++)if(s.charAt(i) != t.charAt(i))min++;
			ans = min(ans, min);
		}
		if(n-zs-1 == zt) {
			int min = 0;
			for(int i = 0; i<n; i++)if(s.charAt(i) == t.charAt(i))min++;
			ans = min(ans, min);
		}
		if(ans == n+1)ans = -1;
		System.out.println(ans);

	}

	//=============================================================================
	//--------------------------The End---------------------------------
	//=============================================================================
	private static long INF = 2000000000000000000L, M = 1000000007, MM = 998244353;
	private static int N = 0;

	

	static FastScanner sc;
	static FastWriter out;

	public static void main(String[] args) throws IOException {
		boolean oj = true;
		if (oj) {
			sc = new FastScanner();
			out = new FastWriter(System.out);
		} else {
			sc = new FastScanner("input.txt");
			out = new FastWriter("output.txt");
		}
		long s = System.currentTimeMillis();
		int t = 1;
		t = sc.nextInt();
		int TTT = 1;
		while (t-- > 0) {
			//			google(TTT++);
			process();
		}
		out.flush();
		//		tr(System.currentTimeMillis()-s+"ms");
	}

	private static boolean oj = System.getProperty("ONLINE_JUDGE") != null;

	

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	

	

	

	

	

	

	

	

	

	

	//custom multiset (replace with HashMap if needed)
	

	

	// compress Big value to Time Limit
	

	// Fast Writer 

	

	// Fast Inputs
	

}
