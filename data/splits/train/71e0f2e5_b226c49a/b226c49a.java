import java.util.*;
import java.io.*;
import java.math.*;

public class A {

	// -- static variables --- //
	static FastReader		sc	= new FastReader();
	static PrintWriter	out	= new PrintWriter(System.out);
	static int					mod	= (int) 1000000007;

	public static void main(String[] args) throws Exception {

		int t = sc.nextInt();
		while (t-- > 0)
			A.go();
		// out.println();
		out.flush();
	}

	// >>>>>>>>>>>>>>>>>>> Code Starts <<<<<<<<<<<<<<<<<<<< //
	static class pair {
		int x, y;

		pair(int x, int y) {
			this.x = x;
			this.y = y;
		}

		

		
	}
	static void go() throws Exception {
		int n=sc.nextInt();
		ArrayList<ArrayList<Integer>> aa=new ArrayList<>();
		for(int i=0;i<n;i++) {
			aa.add(new ArrayList<>());
		}
		LinkedHashMap<pair,Integer> map=new LinkedHashMap<>();
		for(int i=0;i<n-1;i++) {
			int u=sc.nextInt()-1,v=sc.nextInt()-1;
			aa.get(u).add(v);
			aa.get(v).add(u);
			if(u>v) {
				int temp=v;
				v=u;u=temp;
			}
			map.put(new pair(u,v),0);
		}
		
		for(ArrayList<Integer> i : aa) {
			if(i.size()>2) {
				out.println(-1);
				return;
			}
		}
	

		boolean vis[]=new boolean[n];
		dfs(aa,0,vis,2,map);
		for(pair i : map.keySet()) {
			out.print(map.get(i)+" ");
		}
		out.println();
	}
	static void dfs(ArrayList<ArrayList<Integer>> aa,int src,boolean vis[],int see,LinkedHashMap<pair,Integer> map) {
		vis[src]=true;
		for(int i: aa.get(src)) {
			if(vis[i]==false) {
					int x=src,y=i;
					if(x>y) {
						int temp=x;
						x=y;
						y=temp;
					}
					pair temp=new pair(x,y);
					if(see==2) {
							map.put(temp,5);
							see=5;
					}else {
						map.put(temp,2);
						see=2;
					}
					dfs(aa,i,vis,see,map);
			}
		}
	}
	

	// >>>>>>>>>>> Code Ends <<<<<<<<< //

	// --For Rounding--//
	

	// ----Greatest Common Divisor-----//
	

	// --- permutations and Combinations ---//
	static long	fact[];
	static long	invfact[];

	

	// ---sieve---//
	static int prime[] = new int[1000006];

	// static void sieve() {
	//     Arrays.fill(prime, 1);
	//     prime[0] = 0;
	//     prime[1] = 0;
	//     for (int i = 2; i * i <= 1000005; i++) {
	//         if (prime[i] == 1)
	//             for (int j = i * i; j <= 1000005; j +)= i) {
	//                 prime[j] = 0;
	//             }
	//     }
	// }

	// ---- Manual sort ------//
	

	

	// --- Fast exponentiation ---//
	

	// >>>>>>>>>>>>>>> Fast IO <<<<<<<<<<<<<< //
	
}
