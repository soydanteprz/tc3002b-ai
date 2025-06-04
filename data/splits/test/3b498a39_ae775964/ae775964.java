import java.awt.Point;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.ObjectInputStream.GetField;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.sound.sampled.ReverbType;


public class Edu109 {
	static PrintWriter out;
	static Scanner sc;
	static int[]a;
	static int A,B,n,m,h,ans,sum;
	public static void main(String[]args) throws IOException {
		sc=new Scanner(System.in);
		out=new PrintWriter(System.out);
		D();
		out.close();
	}
	  
		
		
		
		
		
		
		static long[][]mem;
		static int ones;
		static ArrayList<Integer>pos;
		static void D() throws IOException {
			int t=1;
			while(t-->0) {
				n=ni();
				a=nai(n);
				mem=new long[n][n];
				ones=0;
				pos=new ArrayList<Integer>();
				for(int i=0;i<n;i++) {
					Arrays.fill(mem[i], -1);
					if(a[i]==1)pos.add(i);
				}
				ones=pos.size();
				long ans=solve(0,0);
				out.println(ans);
			}
		}
		private static long solve(int i, int j) {
			if(i==n||j>=ones)return j==ones?0:(long)1e14;
			if(mem[i][j]!=-1)return mem[i][j];
			long lv=solve(i+1,j);
			if(a[i]==0) {
				int pr=Math.abs(i-pos.get(j));
				lv=Math.min(lv, pr+solve(i+1,j+1));
			}
			return mem[i][j]=lv;
		}
		
		
		
		
	

	
	
	static int ni() throws IOException {
		return sc.nextInt();
	}
	
	
	
	static int[] nai(int n) throws IOException {
		int[] a = new int[n];
		for (int i = 0; i < n; i++)
			a[i] = sc.nextInt();
		return a;
	}
	
	

	
	
	
	
	

	
}
