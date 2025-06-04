import java.util.*;
import java.io.*;
public class C {
    static class Scan {
        private byte[] buf=new byte[1024];
        private int index;
        private InputStream in;
        private int total;
        public Scan()
        {
            in=System.in;
        }
        public int scan()throws IOException
        {
            if(total<0)
            throw new InputMismatchException();
            if(index>=total)
            {
                index=0;
                total=in.read(buf);
                if(total<=0)
                return -1;
            }
            return buf[index++];
        }
        public int scanInt()throws IOException
        {
            int integer=0;
            int n=scan();
            while(isWhiteSpace(n))
            n=scan();
            int neg=1;
            if(n=='-')
            {
                neg=-1;
                n=scan();
            }
            while(!isWhiteSpace(n))
            {
                if(n>='0'&&n<='9')
                {
                    integer*=10;
                    integer+=n-'0';
                    n=scan();
                }
                else throw new InputMismatchException();
            }
            return neg*integer;
        }
        
        
        private boolean isWhiteSpace(int n)
        {
            if(n==' '||n=='\n'||n=='\r'||n=='\t'||n==-1)
            return true;
            return false;
        }
    }
    
    
    
    
    
    
    
    
    static int n,l[],r[];
    static ArrayList<Integer> adj_lst[];
    static long dp[][];
    
    
    public static void main(String args[]) throws IOException {
        Scan input=new Scan();
        StringBuilder ans=new StringBuilder("");
        int test=input.scanInt();
        for(int tt=1;tt<=test;tt++) {
            n=input.scanInt();
            l=new int[n];
            r=new int[n];
            adj_lst=new ArrayList[n];
            for(int i=0;i<n;i++) {
                adj_lst[i]=new ArrayList<>();
                l[i]=input.scanInt();
                r[i]=input.scanInt();
            }
            for(int i=0;i<n-1;i++) {
                int u=input.scanInt()-1;
                int v=input.scanInt()-1;
                adj_lst[u].add(v);
                adj_lst[v].add(u);
            }
            
            dp=new long[n+5][5];
            
            for(int i=0;i<dp.length;i++) {
                for(int j=0;j<dp[0].length;j++) {
                    dp[i][j]=-1;
                }
            }
            
            ans.append(solve(0,-1,0)+"\n");
            
        }
        System.out.println(ans);
    }
    public static long solve(int root,int par,int state) {
        
        if(dp[root+1][state]!=-1) {
            return dp[root+1][state];
        }
        
        long sum1=0,sum2=0;
        
        if(par!=-1 && state==0) {
            sum1+=Math.abs(l[root]-l[par]);
            sum2+=Math.abs(r[root]-l[par]);
        }
        else if(par!=-1) {
            sum1+=Math.abs(l[root]-r[par]);
            sum2+=Math.abs(r[root]-r[par]);
        }
        
        for(int i=0;i<adj_lst[root].size();i++) {
            if(adj_lst[root].get(i)==par) {
                continue;
            }
            sum1+=solve(adj_lst[root].get(i),root,0);
            sum2+=solve(adj_lst[root].get(i),root,1);
        }
        
        long sum=Math.max(sum1,sum2);
        
        dp[root+1][state]=sum;
        
        return sum;
        
    }
}
