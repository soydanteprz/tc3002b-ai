import java.util.*;

import java.io.*;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
	
public class Main {
		private static FS sc = new FS();
		
		
		
		
		
		static int mod = (int) 1e9;
		
		static LinkedList<Integer>[] temp, idx;
		static long inf = (long) Long.MAX_VALUE;
//		static long inf = Long.MAX_VALUE;
//		static int max;
		
		public static void main(String[] args) { 
			int t = sc.nextInt();
//			int t = 1;
			StringBuilder ret = new StringBuilder();
			while(t-- > 0) {
				int n = sc.nextInt();
				String a = sc.next(), b = sc.next();
				int _00 = 0, _01 = 0, _10 = 0, _11 = 0;
				for(int i = 0; i < n; i++) {
					if(a.charAt(i) == '0' && b.charAt(i) == '0') _00++;
					if(a.charAt(i) == '0' && b.charAt(i) == '1') _01++;
					if(a.charAt(i) == '1' && b.charAt(i) == '0') _10++;
					if(a.charAt(i) == '1' && b.charAt(i) == '1') _11++;
				}
				int ans = Integer.MAX_VALUE;
				if(_10 == _01) ans = _01 + _10;
				if(_10 > 0) {
					int n00 = _10 - 1;
		            int n01 = _11;
		            int n10 = _00 + 1;
		            int n11 = _01;
		            if (n01 == n10) {
		                ans = Math.min(ans, 1 + n01 + n10);
		            }
				}
				if(_11 > 0) {
					int n00 = _10;
		            int n01 = _11 - 1;
		            int n10 = _00;
		            int n11 = _01 + 1;
		            if (n01 == n10) {
		                ans = Math.min(ans, 1 + n01 + n10);
		            }
				}
				ret.append(ans == Integer.MAX_VALUE ? -1 : ans);
				ret.append("\n");
			}
			System.out.println(ret);
		}

	}
