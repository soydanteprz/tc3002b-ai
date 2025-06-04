import java.io.BufferedReader;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import java.io.*;

public class abc {
	static PrintWriter pw;

	static long x = 1, y = 1;

	/*
	 * static long inv[]=new long[1000001]; static long dp[]=new long[1000001];
	 */
	/// MAIN FUNCTION///
	public static void main(String[] args) throws Exception {

		FastReader sc = new FastReader();
		pw = new PrintWriter(System.out);
		// use arraylist as it uses the concept of dynamic table(amortized analysis)
		// Arrays.stream(array).forEach(a -> Arrays.fill(a, 0));
		/* List<Integer> l1 = new ArrayList<Integer>(); */
		// Random rand = new Random();
		int tst = sc.nextInt();
		while(tst-->0) {
			int n=sc.nextInt();
			int app[]=new int[n];
			int h[]=new int[n];
			for(int i=0;i<n;i++)
			{
				app[i]=sc.nextInt();
			}
			for(int i=0;i<n;i++)
			{
				h[i]=sc.nextInt();
			}
			long man = 0;
			long last = app[n - 1] - h[n - 1] + 1;
			int end = n-1;
			for (int i = n-2; i >=0; i--) {
				if(app[i]>=last) {
					last = Math.min(last,app[i] - h[i] + 1);
				}
				else {
					long s = app[end]-last+1;
					man += (s*(s+1))/2;
					end = i;
					last = app[i] - h[i] + 1;;
				}
			}
			long s = app[end]-last+1;
			man += (s*(s+1))/2;
			pw.println(man);
		}
		pw.flush();
	}

	

	

	

	

	

	

	

	

	// ch.charAt(i)+"" converts into a char sequence
	

	

	

	

	

	

	

	

	// modular exponentiation
	

	

	/*
	 * public static void extract(ArrayList<Integer> ar, int k, int d) { int c = 0;
	 * for (int i = 1; i < k; i++) { int x = 0; boolean dm = false; while (x > 0) {
	 * long dig = x % 10; x = x / 10; if (dig == d) { dm = true; break; } } if (dm)
	 * ar.add(i); } }
	 */

	

// counts the set(1) bit of a number
	

//tells whether a particular index has which bit of a number	
	

	

	

//use collections.swap for swapping 
	

	

	

	// return the hash value of a string
	

}
