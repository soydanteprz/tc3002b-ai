import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.InputMismatchException;
import java.util.*;
import java.io.*;
import java.lang.*;
public class Main{
    
 
    
    
    public static void sortbyColumn(int arr[][], int col) 
    { 
        
        Arrays.sort(arr, new Comparator<int[]>() { 
            
           
        });  
    } 
    
    public static class Pair<U extends Comparable<U>, V extends Comparable<V>> implements Comparable<Pair<U, V>> {
        public U x;
        public V y;
    
        public Pair(U x, V y) {
            this.x = x;
            this.y = y;
        }
    
        
    
        
    
        
    
        
    
    }
    
    
    
    
    
    
    
    // static void calc(int n){
    //     int i,j;
    //     dp[n-1]=0;
    //     if(n>1)
    //     dp[n-2]=1;
    //     for(i=n-3;i>=0;i--){
    //         long ind=n-i-1;
    //         dp[i]=((ind*(long)mypow(10,ind-1))%MOD+dp[i+1])%MOD;
    //     }
    // }
    
    static long dist[],dp[][],left[],right[];
    static int visited[],isit[];
    static ArrayList<Pair<Integer,Pair<Long,Long>>> adj[],li;
    //static int dp[][][];
    static int MOD=1000000007;
    static char ch[];
    static int[] sieve,is_sieve;
    static TreeSet<Integer> tr;
    static long mat[][];
    
    
    static void dfs(int node,int par, Pair<Long,Long> p[]){
        for(Pair<Integer,Pair<Long,Long>> pp:adj[node]){
            if(pp.x!=par){
                //sum+=Math.abs(selected[node]-selected[pp.x]);
                dfs(pp.x,node,p);
                //System.out.println(node+" "+pp.x);
                long x=Math.abs(p[node].x-p[pp.x].x);
                long y=Math.abs(p[node].x-p[pp.x].y);
                long z=Math.abs(p[node].y-p[pp.x].x);
                long w=Math.abs(p[node].y-p[pp.x].y);
                
                left[node]+=Math.max(x+left[pp.x],y+right[pp.x]);
                right[node]+=Math.max(z+left[pp.x],w+right[pp.x]);
                
            }
        }
    }
    
    public static void main(String args[]){
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter w = new PrintWriter(outputStream);
        int t,i,j,tno=0,tte;
        
        t=in.nextInt();
        //t=1;
        //tte=t;
        
        while(t-->0){
            //sum=0;
            int n=in.nextInt();
            adj=new ArrayList[n+1];
            left=new long[n+1];
            right=new long[n+1];
            visited=new int[n+1];
            for(i=0;i<n+1;i++){
                adj[i]=new ArrayList<>();
            }
            Pair<Long,Long> p[]=new Pair[n+1];
            for(i=1;i<=n;i++){
                p[i]=new Pair<>(in.nextLong(),in.nextLong());
            } 
            for(i=0;i<n-1;i++){
                int u,v;
                u=in.nextInt();
                v=in.nextInt();
                adj[u].add(new Pair<>(v,p[v]));
                adj[v].add(new Pair<>(u,p[u]));
            }
            
            dfs(1,-1,p);
            // for(i=0;i<n+1;i++){
            //     w.print(selected[i]+" ");
            // }
            // w.println();
            w.println((long)Math.max(left[1],right[1]));
        }
        w.close();
    }
}
