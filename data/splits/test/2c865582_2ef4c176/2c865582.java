import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.StringTokenizer;

public class Main {
    


    static long dfs(int cur,int parent,int taken,HashMap<Integer,ArrayList<node>> hm,ArrayList<node> al,long[][] dp){
        long ans=0;
        if(dp[cur-1][taken]!=-1) return dp[cur-1][taken];
        int v=taken==0?al.get(cur-1).l:al.get(cur-1).r;
        for(node x:hm.get(cur)){
            int xl=x.l,xr=x.r,idx=x.idx;
            if(idx!=parent){
                ans+=Math.max(Math.abs(v-xl)+dfs(idx,cur,0,hm,al,dp),Math.abs(v-xr)+dfs(idx,cur,1,hm,al,dp));
            }
        }
        return dp[cur-1][taken]=ans;
    }
    public static void main(String[] args) {
        FastReader obj = new FastReader();
        int t = obj.nextInt();
        while(t-->0) {
            int n = obj.nextInt(),a,b;
            ArrayList<node> al=new ArrayList<>();
            for(int i=0;i<n;i++){
                a = obj.nextInt();
                b = obj.nextInt();
                al.add(new node(a,b,i+1));
            }
            HashMap<Integer,ArrayList<node>> hm=new HashMap<>();
            for(int i=0;i<n-1;i++){
                a = obj.nextInt();
                b = obj.nextInt();
                if(!hm.containsKey(a)){
                    hm.put(a,new ArrayList<node>());
                }
                if(!hm.containsKey(b)){
                    hm.put(b,new ArrayList<node>());
                }
                hm.get(a).add(al.get(b-1));
                hm.get(b).add(al.get(a-1));
            }
            long[][] dp=new long[n+2][2];
            for(long[] x:dp) Arrays.fill(x,-1);
            long ans1,ans2;
            ans1=dfs(1,0,0,hm,al,dp);
            ans2=dfs(1,0,1,hm,al,dp);
            System.out.println(Math.max(ans1,ans2));
        }
    }
}

class node{
    int l,r,idx;
    node(int l1,int r1,int i){
        l=l1;
        r=r1;
        idx=i;
    }
}
