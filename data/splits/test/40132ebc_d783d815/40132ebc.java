import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.Comparator;

import java.util.PriorityQueue;
import java.util.List;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.HashSet;
import java.util.Stack;
import java.util.Queue;



import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;
import static java.lang.Math.abs;
import static java.lang.Math.random;
import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.MIN_VALUE;
import static java.util.Collections.reverseOrder;

public final class Test1 {
	static int mod = 1000000007;
	static int mod1 = 998244353;
	public static void main(String[] args) {
		solve();
	}
	
	
	
	
	  //only for prime m
	
	
	
	
	
	
	
	
	

	

	





//	<------------------------- CODE START FROM HERE -------------------------------->


	public static void solve() {
		FastScanner sc = new FastScanner();
		int t = sc.nextInt();
		while (t-- != 0) {
			int n = sc.nextInt();
			String[] arr = new String[n];
			for (int i = 0; i < n; i++)
				arr[i] = sc.next();

			int cnt[] = new int[5];
			for (int i = 0; i < n; i++) {

				for (char ch : arr[i].toCharArray()) {
					cnt[ch - 'a']++;
				}
			}

			int res = 0;
			for (int i = 0; i < 5; i++) {

				res = Math.max(res, fun(arr, cnt, (char)( 'a' + i)));
			}
			System.out.println(res);

		}
	}
	static int fun(String[] arr, int[] cnt, char c) {
		int n = arr.length;

		int total = cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] - cnt[c - 'a'];
		int letter_total = cnt[c - 'a'];


		PriorityQueue<Pair>pq = new PriorityQueue<>();

		for (int i = 0; i < n; i++) {

			int letter = 0;
			int ext = 0;
			for (char ch : arr[i].toCharArray()) {

				if (ch == c)
					letter++;
				else
					ext++;
			}
			pq.offer(new Pair(letter, ext));
		}



		while (pq.size() > 0) {

			if (total < letter_total)
				return pq.size();


			Pair temp = pq.poll();

			total -= temp.ext;
			letter_total -= temp.letter;
		}


		return 0;
	}

	static class Pair implements Comparable<Pair> {
		int letter, ext;
		Pair(int letter, int ext) {
			this.letter = letter;
			this.ext = ext;
		}
		
	}











// <----------------------------------CODE END HERE------------------------>

// System.out.println();
// int b[] = a.clone();


}
