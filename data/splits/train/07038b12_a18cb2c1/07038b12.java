import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

public class h {

	public static int r1=0;
	 
	static long[][] f=new long[501][501];
	public static void main(String[] args)  {
		OutputStream outputStream =System.out;
	    PrintWriter out =new PrintWriter(outputStream);
		FastReader s=new FastReader();
			
		   int t=s.nextInt();
		 while(t>0) {
				int n=s.nextInt();
				for(int i=1;i<=n;i++) {
					vis[i]=0;
					deg[i]=0;
					ans[i]=0;
					f1[i]=new ArrayList<>();
				}
				for(int i=1;i<=n-1;i++) {
					int x=s.nextInt();
					int y=s.nextInt();
					f1[x].add(new pair1(y,i));
					f1[y].add(new pair1(x,i));
					deg[x]++;
					deg[y]++;
				}
//				dfs(1);
				int node=-1;
				boolean p=false;
				for(int i=1;i<=n;i++) {
					if(deg[i]>2) {
						p=true;
					}
					if(deg[i]==1) {
						node=i;
					}
				}
				if(p) {
					out.println(-1);
				}else {
					dfs(node,2);
//					System.out.println();
				for(int i=1;i<=n-1;i++) {
					out.print(ans[i]+" ");
				}
				out.println();
				}
			 t--;
			   }
			
		
		
	out.close();
	}
	static int[] ans=new int[100001];
	static int[] deg=new int[100001];
	static void dfs(int node,int v) {
		vis[node]=1;
//		deg[node]++;

		for(int i=0;i<f1[node].size();i++) {
			if(vis[f1[node].get(i).a]==0) {
			int g=2;
			if(v==2) {
				g=5;
			}
			ans[f1[node].get(i).b]=v;
				dfs(f1[node].get(i).a,g);
//				deg[node]++;
				
			}
			
		}
	}
//	static long ans(int[] a,int[] b,long till,long[] post,int n,int k,int i) {
//		if(k==0 || i==n) {
//			return till+post[i];
//		}
//		if(f[i][k]!=-1) {
//			return f[i][k];
//		}
//		
//	}
//	static class pair
	static int[] vis=new int[100001];
	static class pair1 {
		private int a;
		private int b;
		pair1(int a,int b){
			this.a=a;
			this.b=b;
		}
		
		
		
	}
	
	
	
	public static int[] is_prime=new int[100001];
	public static ArrayList<Long> primes=new ArrayList<>();
	
	
	static ArrayList<pair1>[] f1=new ArrayList[100001];
	
	
	
	
	
}
