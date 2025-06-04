import java.io.*;
import java.util.*;
public class D_668 {
    static int p=1000000007;
    static  int dia=Integer.MIN_VALUE;
    public static void main(String[] args) throws Exception{
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(java.io.FileDescriptor.out), "ASCII"), 512);
        FastReader sc=new FastReader();
        int t=sc.nextInt();
       StringBuilder sb=new StringBuilder();
        while(t-->0)
        {
            int n=sc.nextInt();
            int a=sc.nextInt();
            int b=sc.nextInt();
            int da=sc.nextInt();
            int db=sc.nextInt();
            dia=Integer.MIN_VALUE;
            int dp[]=new int[(int) n+1];
            List<Integer> adj[]=new ArrayList[n+1];
            for(int i=0;i<=n;i++)
            {
                adj[i]=new ArrayList<>();
            }
            Set<Integer> s=new HashSet<>();
            for(int i=1;i<n;i++)
            {
                int u=sc.nextInt();
                int v=sc.nextInt();
                adj[u].add(v);
                adj[v].add(u);
            }
            dfs(dp,adj,a,-1,0);
           // System.out.println(dia+" "+dp[b]);
            if(2*da>=db||2*da>=dia||da>=dp[b])
                sb.append("Alice\n");
            else
                sb.append("Bob\n");

        }
        System.out.println(sb.toString());
        out.flush();
    }
    public static int dfs(int dp[],List<Integer> adj[],int i,int p,int d)
    {
        dp[i]=d;
        int last_max=0;
        for(int v:adj[i])
        {
            if(p!=v&&dp[v]==0)
            {
                int max=1+dfs(dp,adj,v,i,d+1);
                dia=Math.max(dia,max+last_max);
                last_max=Math.max(last_max,max);
            }
        }
        return last_max;
    }
    
    
    int bit[]=new int[(int)1e6];
    
    


    ///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
    


}
