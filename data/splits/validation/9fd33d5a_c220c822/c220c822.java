import java.util.*;
import java.io.*;
import java.lang.*;
public class Main{

    public static void main(String args[]){
        InputReader in=new InputReader(System.in);
        TASK solver = new TASK();
        int t=1;
        t = in.nextInt();
        for(int i=1;i<=t;i++)
        {
            solver.solve(in,i);
        }
    }
    static class TASK {
        static int dp[] = new int[31];
        static {
            dp[0]=1;
            for(int i=1;i<31;i++)
            {
                dp[i]=dp[i-1]*2;
            }
        }
        void solve(InputReader in, int testNumber) {
            int n = in.nextInt();
            int k = in.nextInt();
            int a[] = new int[k];
            int l[] = new int[n+1];
            long pre[] = new long[n+2];
            long suff[] = new long[n+2];
            Arrays.fill(pre,Long.MAX_VALUE/2);
            Arrays.fill(suff,Long.MAX_VALUE/2);
            for(int i=0;i<k;i++)
            {
                a[i]=in.nextInt();
            }
            for(int i=0;i<k;i++)
            {
                int x = in.nextInt();
                l[a[i]]=x;
            }
            for(int i=1;i<=n;i++)
            {
                pre[i]=pre[i-1];
                if(l[i]!=0)
                {
                    pre[i]=Math.min(pre[i],l[i]-i);
                }
            }
            for(int i=n;i>=1;i--)
            {
                suff[i]=suff[i+1];
                if(l[i]!=0)
                {
                    suff[i]=Math.min(suff[i],l[i]+i);
                }
            }
            for(int i=1;i<=n;i++)
            {
                System.out.print(Math.min(pre[i]+i,suff[i]-i)+" ");
            }
            System.out.println();







        }
    }


    

    
    
}
