import java.math.BigInteger;
//import static java.lang.Math.max;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Vector;

import java.util.Scanner;

public class ahh {
//trihund
	static Scanner scn = new Scanner(System.in);
	static boolean vis[][];

	

	static FastReader s = new FastReader();
	static int MOD = 1000000007;

	public static void main(String[] args) {
		
		int n=scn.nextInt(),count=0;
		int arr[]=new int[n];
		for(int i=0;i<n;i++)
		{
			arr[i]=scn.nextInt();
		}
		ArrayList<Integer>zer=new ArrayList<Integer>(),one=new ArrayList<Integer>();
		for(int i=0;i<n;i++)
		{
			if(arr[i]==0)
				zer.add(i);
			else
				one.add(i);
		}
		count=one.size();
		long memo[][]=new long[one.size()+1][zer.size()+1];
		for(int i=0;i<=one.size();i++)
		{
			for(int j=0;j<=zer.size();j++)
				memo[i][j]=-1;
		}
		System.out.println(arm(one, zer, 0, 0, count,memo));
				
		
	}

	public static long arm(ArrayList<Integer>one,ArrayList<Integer>zer,int i,int j,int count,long memo[][])
	{	if(count==0)
		return 0;
		if(i==one.size()||j==zer.size())
			return Integer.MAX_VALUE;
		
		if(memo[i][j]!=-1)
			return memo[i][j];
		
		long a=Integer.MAX_VALUE,b=Integer.MAX_VALUE;
		a=arm(one, zer, i+1, j+1,count-1,memo)+Math.abs(one.get(i)-zer.get(j));
		b=arm(one, zer, i, j+1,count,memo);
		memo[i][j]=Math.min(a, b);
		return Math.min(a, b);
	}
	

	

	

}








