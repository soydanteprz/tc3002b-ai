import java.util.*;
import java.io.*;
import java.lang.*;
public class Main{

    public static void main(String args[]){
        InputReader in=new InputReader(System.in);
        TASK solver = new TASK();
        int t=1;
//        t = in.nextInt();
        for(int i=1;i<=t;i++)
        {
            solver.solve(in,i);
        }
    }
    static class TASK {
        static int mod = 1000000007;
        static long solve(int a[],ArrayList<Integer> al,int i,int j,long dp[][])
        {
            if(j<0 || i<0)
                return Integer.MAX_VALUE;
            if(i==0 && j==0)
                return 0;
            if(j==0)
                return 0;
            if(i==0)
                return Integer.MAX_VALUE;
            if(dp[i-1][j-1]==-1)
            {
                dp[i-1][j-1]=solve(a,al,i-1,j-1,dp);
            }
            if(dp[i-1][j]==-1)
                dp[i-1][j]=solve(a,al,i-1,j,dp);
            if(a[i-1]==1)
                return dp[i-1][j];
            dp[i][j]=Math.min(dp[i-1][j],dp[i-1][j-1]+Math.abs(i-al.get(j-1)));
            return dp[i][j];
        }
        void solve(InputReader in, int testNumber) {
            int n = in.nextInt();
            int a[] = new int[n+1];
            long dp[][] = new long[n+1][n+1];
            ArrayList<Integer> al = new ArrayList<>();
            for(int i=0;i<n;i++) {
                a[i] = in.nextInt();
                if(a[i]==1)
                    al.add(i+1);
                Arrays.fill(dp[i],-1);
            }
            Arrays.fill(dp[n],-1);
            System.out.println(solve(a,al,n,al.size(),dp));










        }
    }

    

    
    
}
