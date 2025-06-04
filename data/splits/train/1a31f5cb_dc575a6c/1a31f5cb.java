import java.util.*;
import java.io.*;

public class Solution
{
    
    
    static void dfs(int n , LinkedList<Integer> g[] , int arr[][] , int p , long dp[][])
    {
        for(Integer i : g[n])
        {
            if(i != p)
            {
                dfs(i,g,arr,n,dp);
            dp[n][0] += Math.max(dp[i][0]+Math.abs(arr[n][0]-arr[i][0]) , 
                       dp[i][1]+Math.abs(arr[n][0]-arr[i][1]));
                       
                 
            dp[n][1] += Math.max(dp[i][0]+Math.abs(arr[n][1]-arr[i][0]) , 
                       dp[i][1]+Math.abs(arr[n][1]-arr[i][1]));
            }                
        }
    }
    public static void main(String []args) throws IOException
    {
        Reader sc = new Reader();
        StringBuffer str = new StringBuffer("");
        int t  = sc.nextInt();
        while(t-- > 0)
        {
            int n = sc.nextInt();
            int arr[][] = new int[n][2];
            for(int i = 0 ; i < n;  i++)
            {
                arr[i][0] = sc.nextInt();
                arr[i][1] = sc.nextInt();
            }
            LinkedList<Integer> g[] = new LinkedList[n];
            for(int i = 0 ; i < n ; i++)
            {
                g[i] = new LinkedList<Integer>();
            }
            for(int i = 0 ; i < n-1 ; i++)
            {
                int x = sc.nextInt()-1;
                int y = sc.nextInt()-1;
                g[x].add(y);
                g[y].add(x);
            }
            long dp[][] = new long[n][2];
            dfs(0,g,arr,-1 , dp);
            str.append(Math.max(dp[0][0] , dp[0][1]));
            str.append(System.lineSeparator());
        }
        System.out.println(str);
    }
}
