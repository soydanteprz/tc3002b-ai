import java.util.*;
import java.lang.*;
import java.io.*;

public class Codeforces {
	
	public static void main (String[] args) throws java.lang.Exception
	{
		// your code goes here
		FastReader sc=new FastReader();
		int n=sc.nextInt();
		int a[]=new int[n];
		ArrayList<Integer> arr0=new ArrayList<>();
		ArrayList<Integer> arr1=new ArrayList<>();
		for(int i=0;i<n;i++)
		{
		    a[i]=sc.nextInt();
		    if(a[i]==0)
		    arr0.add(i);
		    else
		    arr1.add(i);
		}
		n=arr0.size();
		int m=arr1.size();
		int dp[][]=new int[m+1][n+1];
		for(int i=0;i<=n;i++)
		{
		    dp[0][i]=0;
		}
		for(int i=1;i<=m;i++)
		{
		    dp[i][i]=dp[i-1][i-1]+Math.abs(arr0.get(i-1)-arr1.get(i-1));
		    for(int j=i+1;j<=n;j++)
		    {
		        dp[i][j]=Math.min(dp[i-1][j-1]+Math.abs(arr0.get(j-1)-arr1.get(i-1)),dp[i][j-1]);
		    }
		}
		System.out.println(dp[m][n]);
	}
}
