import java.util.*;


import java.io.*;
 public class Solution {
		static FastScanner scr=new FastScanner();
//		static Scanner scr=new Scanner(System.in);
		static PrintStream out=new PrintStream(System.out);
		static StringBuilder sb=new StringBuilder();
		
		
		
		
		
		
		
		
		static HashMap<Character,ArrayList<Integer>>hm=new HashMap<>();
		static void solve() {
			int n=scr.nextInt();
			
			int count[]=new int[5];
			int store[][]=new int[n][5];
			int max=MIN;
			int length[]=new int[n];
			for(int i=0;i<n;i++) {
				String s=scr.next();
				length[i]=s.length();
				for(int j=0;j<s.length();j++) {
					store[i][s.charAt(j)-'a']++;
					count[s.charAt(j)-'a']++;
				}
			}
			
			for(int j=0;j<5;j++) {
				
				ArrayList<Integer>a=new ArrayList<>();
				for(int i=0;i<n;i++) {
					a.add((store[i][j]*2)-length[i]);
				}
				
				Collections.sort(a,Collections.reverseOrder());
				
				long sum=0;
				int ans=0;
				for(int i=0;i<a.size();i++) {
					sum+=a.get(i);
					if(sum>0) {
						ans++;
					}else {
						break;
					}
				}
				max=Math.max(max, ans);
			}
			out.println(max);
			
		}
		static int  MAX = Integer.MAX_VALUE;
		static int  MIN = Integer.MIN_VALUE;
		public static void main(String []args) {
			int t=scr.nextInt();
			while(t-->0) {
				solve();
			}
//			solve();
//			out.println(sb);
		}
	}
