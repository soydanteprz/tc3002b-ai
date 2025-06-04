import java.util.*;
import java.util.Map.Entry;
import java.lang.*;
import java.io.*;
import java.math.BigInteger;


public class CF {
	private static FS sc = new FS();
	
	
	
	
	
	static int mod = 998244353;//(int) 1e9 + 7;
//	static int mod = (int) 998244353;
	static int max = (int) 1e6, sq = 316;
	static LinkedList<Integer>[] temp;
//	static int[] par, rank;
	
	static class pair implements Comparator<pair>{
		int val1;
		int val2;
		int idx;
		pair(int x, int y, int z) {
			val1 = x; val2 = y; idx = z;
		}
		pair() {}
		
		
	}
	
	public static void main(String[] args) {	
		int t = sc.nextInt();
//		int t = 1;
		StringBuilder ret = new StringBuilder();
		while(t-- > 0) {
			int n = sc.nextInt();
			String[] s = new String[n];
			int[][] pre = new int[n][5];
			for(int i = 0; i < n; i++) {
				s[i] = sc.next();
				for(int j = 0; j < s[i].length(); j++) {
					pre[i][s[i].charAt(j)-'a']++;
				}
			}
			int max = 0;
			for(int i = 0; i < 5; i++) {
				int[] vis = new int[n];
				int co = 0, toC = 0, oth = 0;
//				char tar = (char)(i+'a');
				for(int j = 0; j < n; j++) {
					int tot = 0;
					for(int aa:pre[j]) tot += aa;
					int rem = tot-pre[j][i];
					if(pre[j][i] > rem) {
						co++;
						toC += pre[j][i];
						vis[j] = 1;
						oth += rem;
					}
				}
				PriorityQueue<pair> q = new PriorityQueue<>(new pair());
				for(int j = 0; j < n; j++) {
					if(vis[j] == 1) continue;
					int tot = 0;
					for(int aa:pre[j]) tot += aa;
					int rem = tot - pre[j][i];
					q.add(new pair(pre[j][i], rem, j));
				}
//				for(pair aa:q) System.out.println(aa.val1 + " " + aa.idx);
//				System.out.println();
				while(!q.isEmpty()) {
					pair cur = q.poll();
					if(cur.val1 + toC > cur.val2+oth) {
						toC += cur.val1;
						oth += cur.val2;
						co++;
					}
				}
//				System.out.println(co);
				max = Math.max(max, co);
			}
			ret.append(max + "\n");
   		}
		System.out.println(ret);
	}
	
	
}
