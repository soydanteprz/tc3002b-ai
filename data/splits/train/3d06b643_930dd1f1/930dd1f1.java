import java.util.*;
import java.io.*;
public class Armchairs 
{
	public static int val(int b[],int m,int c[],int n,int dp[][])
	{
		if(dp[m][n]>-1)
		{
			return dp[m][n];
		}
		else
		if(m==0)
		{
			return 0;
		}
		else
		if(n==0)
		{
							return 2099999999;
		}
		else
		{
		
			return dp[m][n]=Math.min(val(b,m-1,c,n-1,dp)+Math.abs(b[m-1]-c[n-1]),val(b,m,c,n-1,dp));
		}
	}
	public static void process()throws IOException
	{
		int n=I();
		int a[]=Ai(n);
		ArrayList<Integer> arr=new ArrayList<Integer>();
		ArrayList<Integer> arr1=new ArrayList<Integer>();
		for(int i=0;i<n;i++)
		{
			if(a[i]==0)
			{
				arr1.add(i);
			}
			else
			{
				arr.add(i);
			}
		}
		int b[]=new int[arr.size()];
		int c[]=new int[arr1.size()];
		int m=arr.size();int n1=arr1.size();
		int dp[][]=new int[m+1][n1+1];

		dyn(dp,m+1,n1+1,-1);
		
		for(int i=0;i<arr.size();i++)
		{
			b[i]=arr.get(i);
		}
		for(int i=0;i<arr1.size();i++)
		{
			c[i]=arr1.get(i);
		}

		arr.clear();
	    arr1.clear();
	   pn(val(b,m,c,n1,dp));
	}
	static Scanner sc = new Scanner(System.in);
	static PrintWriter out = new PrintWriter(System.out);
	static void pn(Object o){out.println(o);out.flush();}
	
	
	static int I() throws IOException{return sc.nextInt();}
	
	
	
	
	static int[] Ai(int n) throws IOException{int[] arr = new int[n];for (int i = 0; i < n; i++)arr[i] = I();return arr;}
	
	
	static void dyn(int dp[][],int n,int m,int z)throws IOException {for(int i=0;i<n;i++){ for(int j=0;j<m;j++){ dp[i][j]=z;}} }
	
//	*--------------------------------------------------------------------------------------------------------------------------------*//
	
	public static void main(String[] args)throws IOException{try{boolean oj=true;if(oj==true)
	{AnotherReader sk=new AnotherReader();PrintWriter out=new PrintWriter(System.out);}
	else
	{AnotherReader sk=new AnotherReader(100);out=new PrintWriter("output.txt");}
	//long T=L();while(T-->0)
	{process();}out.flush();out.close();}catch(Exception e){return;}}}
