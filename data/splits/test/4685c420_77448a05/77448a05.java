import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.StringTokenizer;

public class P4 {
	static int N = 1000001;

	// Array to store inverse of 1 to N
	static long[] factorialNumInverse = new long[N + 1];

	// Array to precompute inverse of 1! to N!
	static long[] naturalNumInverse = new long[N + 1];

	// Array to store factorial of first N numbers
	static long[] fact = new long[N + 1];

	// Function to precompute inverse of numbers
	

	// Function to precompute inverse of factorials
	

	// Function to calculate factorial of 1 to N
	

	// Function to return nCr % p in O(1) time
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		FastReader sc = new FastReader();
		int t;
		t = sc.nextInt();
		StringBuilder st = new StringBuilder();
//		int p = 998244353;
//	    InverseofNumber(p);
//	    InverseofFactorial(p);
//	    factorial(p);
		// System.out.println(3|1);
		while (t-- != 0) {
			int n=sc.nextInt();
			String s[]=new String[n];
			for(int i=0;i<n;i++) {
				s[i]=sc.nextLine();
			}
			int max=0;
			ArrayList<Integer> freq=new ArrayList<>();
			for(int i=0;i<5;i++) {
				int f1=0;
				freq=new ArrayList<>();
				for(int j=0;j<n;j++) {
					f1=0;
					for(int k=0;k<s[j].length();k++) {
						if(s[j].charAt(k)==i+'a') {
							f1++;
						}else {
							f1--;
						}
					}
					freq.add(f1);
				}
				Collections.sort(freq);
				int x=0;
				int total=0;
				for(int j=n-1;j>=0;j--) {
					total=total+freq.get(j);
					if(total>0) {
						x++;
					}else {
						break;
					}
				}
				max=Math.max(max, x);
			}
			st.append(max+"\n");
		}
		System.out.println(st);
	}

	static FastReader sc = new FastReader();

	

	static void sort(int[] A) {
		int n = A.length;
		Random rnd = new Random();
		for (int i = 0; i < n; ++i) {
			int tmp = A[i];
			int randomPos = i + rnd.nextInt(n - i);
			A[i] = A[randomPos];
			A[randomPos] = tmp;
		}
		Arrays.sort(A);
	}

	static void sort(long[] A) {
		int n = A.length;
		Random rnd = new Random();
		for (int i = 0; i < n; ++i) {
			long tmp = A[i];
			int randomPos = i + rnd.nextInt(n - i);
			A[i] = A[randomPos];
			A[randomPos] = tmp;
		}
		Arrays.sort(A);
	}

	static String sort(String s) {
		Character ch[] = new Character[s.length()];
		for (int i = 0; i < s.length(); i++) {
			ch[i] = s.charAt(i);
		}
		Arrays.sort(ch);
		StringBuffer st = new StringBuffer("");
		for (int i = 0; i < s.length(); i++) {
			st.append(ch[i]);
		}
		return st.toString();
	}

	

	
}
