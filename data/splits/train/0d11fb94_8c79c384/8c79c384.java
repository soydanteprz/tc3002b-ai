import java.util.*;

import javax.print.attribute.HashAttributeSet;


import java.io.*;
import java.math.BigInteger;
import java.sql.Array;
public class CP {
	  
	 static int mod  = 1000000007;
	 
	 
	 
	 
	 	
	 
	 

/* ***************************************************************************************************************************************************/	 
	 static FR sc = new FR();
	 static StringBuilder sb = new StringBuilder();
	 public static void main(String args[]) {
//		 int tc = sc.nextInt();
//		 while(tc-- > 0) {
			 TEST_CASE();
//		 }
			System.out.println(sb);
		 }		

		static void TEST_CASE() {
			int n = sc.nextInt();
			int[] arr = new int[n];
			for(int i = 0 ; i< n ;i++) {
				arr[i] = sc.nextInt();
			}
			
			ArrayList<Integer> ao = new ArrayList<Integer>();
			ArrayList<Integer> az = new ArrayList<Integer>();
			for(int i = 0 ; i< n ;i++) {
				if(arr[i] == 1) ao.add(i);
				else az.add(i);
			}
			long[][] dp = new long[n+1][n+1];
			for(int i = 0 ; i<n ; i++) Arrays.fill(dp[i], -1);
			
			sb.append(fnc(dp, ao, az, 0, 0));
			
		}
		static long fnc(long[][] dp ,ArrayList<Integer> ao , ArrayList<Integer> az ,int i , int j) {
			if(i == ao.size()) return 0;
			if(j == az.size()) return Long.parseLong("1000000000000");
			if(dp[i][j] != -1) return dp[i][j];
			long a = Math.abs(ao.get(i) - az.get(j)) + fnc(dp, ao, az, i+1, j+1);
			long b = fnc(dp, ao, az, i, j+1);
			dp[i][j] = Math.min(a, b);
			return dp[i][j];
			
		}
		
}
