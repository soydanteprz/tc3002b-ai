import java.util.*;
import java.io.*;
import java.math.BigInteger;
import java.sql.Array;
public class CP {
	  
	 static int mod  = 1000000007;
	 
	 
	 
	
	 

/* ***************************************************************************************************************************************************/	 
	 static FR sc = new FR();
	 static StringBuilder sb = new StringBuilder();
	 public static void main(String args[]) {
		 int tc = sc.nextInt();
	
		 while(tc-- > 0) {
			 TEST_CASE();
		 }
			System.out.println(sb);
		 }		

		static void TEST_CASE() {
		int n = sc.nextInt();
		 lr = new long[n][2];
		for(int  i =0 ; i < n ; i ++) {
			lr[i][0] = sc.nextLong();
			lr[i][1] = sc.nextLong();
		}
		 adj = new ArrayList<ArrayList<Integer>>();
		for(int i = 0 ; i <n ; i++) adj.add(new ArrayList<Integer>());
		
		for(int i = 0 ; i<n-1 ; i++) {
			int u = sc.nextInt()-1 , v = sc.nextInt()-1;
			adj.get(u).add(v);
			adj.get(v).add(u);
		}
		min = new long[n];
		max = new long[n];
		dfs(0,-1);
		sb.append(Math.max(min[0], max[0])).append("\n");
		
		}
		static long[] min , max , lr[];
		static ArrayList<ArrayList<Integer>> adj;
		static void dfs(int u , int p  ) {
			
			for(int child:adj.get(u)) {
				if(child == p) continue;
				dfs(child , u);
			}
			long left = lr[u][0] , right = lr[u][1];
			
			long ansl = 0 , ansr = 0;
			for(int child:adj.get(u)) {
				if(child == p) continue;
				long leftc = lr[child][0] , rightc = lr[child][1];
				ansl += Math.max( min[child] + Math.abs(left - leftc)  ,
						max[child] +Math.abs(left - rightc)      );
			}
			for(int child:adj.get(u)) {
				if(child == p) continue;
				long leftc = lr[child][0] , rightc = lr[child][1];
				ansr += Math.max( min[child] + Math.abs(right - leftc)  ,
						max[child] +Math.abs(right - rightc)      );
			}
			min[u] = ansl;
			max[u] = ansr;
			
		}
		
}
