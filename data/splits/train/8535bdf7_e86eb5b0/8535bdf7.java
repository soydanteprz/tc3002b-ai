import java.util.*;
import java.io.*;
import java.lang.*;
public class Problem {
    public static void dfs(int u, int parent, ArrayList<ArrayList<Integer>> graph,long[][] dp,Pair[] arr)
    {
        ArrayList<Integer> adja=graph.get(u);
        for(int i=0;i<adja.size();i++)
        {
            int adja_ele=adja.get(i);
            if(adja_ele!=parent)
            {
                dfs(adja_ele, u,graph,dp,arr);
                long res1=Math.max(Math.abs(arr[u].getL()-arr[adja_ele].getL())+dp[adja_ele][0],Math.abs(arr[u].getL()-arr[adja_ele].getR())+dp[adja_ele][1]);
                long res2=Math.max(Math.abs(arr[u].getR()-arr[adja_ele].getL())+dp[adja_ele][0],Math.abs(arr[u].getR()-arr[adja_ele].getR())+dp[adja_ele][1]);
                dp[u][0]+=res1;
                dp[u][1]+=res2;
//                System.out.println(u+" "+adja_ele+" "+res1+" "+res2);

            }
        }
    }



    public static void main(String[] args) throws IOException{
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
        int t =Integer.parseInt(br.readLine());
        while(t-- >0)
        {

            int n =Integer.parseInt(br.readLine().trim());
            Pair[] arr=new Pair[n+1];

            for(int i=1;i<n+1;i++) {
                String[] sa=br.readLine().trim().split(" ");
                int l=Integer.parseInt(sa[0]);
                int r=Integer.parseInt(sa[1]);
                Pair p=new Pair(l,r);
                arr[i]=p;
            }
            ArrayList<ArrayList<Integer>> graph=new ArrayList<ArrayList<Integer>>(n+1);
            for(int i=0;i<n+1;i++)
            {
               graph.add(new ArrayList<Integer>());
            }
            for(int i=0;i<n-1;i++)
            {
                String[] sa=br.readLine().trim().split(" ");
                int u=Integer.parseInt(sa[0]);
                int v=Integer.parseInt(sa[1]);
                graph.get(u).add(v);
                graph.get(v).add(u);
            }
            long[][] dp=new long[n+1][2];
            dfs(1,0,graph,dp,arr);
//            for(int i=0;i<dp.length;i++)
//            {
//                System.out.println(dp[i][0]+" "+dp[i][1]);
//            }
            System.out.println(Math.max(dp[1][0],dp[1][1]));
        }
    }

}
class Pair
{
    private int l;
    private int r;

    public Pair(int l, int r) {
        this.l = l;
        this.r = r;
    }

    public int getL() {
        return l;
    }

    

    public int getR() {
        return r;
    }

    
}
