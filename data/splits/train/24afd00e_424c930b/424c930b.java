import java.io.*;
import java.util.*;


public class Solution{
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Task solver = new Task();
        solver.solve(in, out);
        out.close();
    }
    // main solver
    static class Task{

        double eps= 0.00000001;
        static final int MAXN = 10000001;

        // stores smallest prime factor for every number
        static int spf[] = new int[MAXN];

        Map<Integer,Set<Integer>> dp= new HashMap<>();

        // Calculating SPF (Smallest Prime Factor) for every
        // number till MAXN.
        // Time Complexity : O(nloglogn)
        

        // A O(log n) function returning primefactorization
        // by dividing by smallest prime factor at every step
        
        // function to find first index >= x
        

        // function to find last index <= y
        

        // function to count elements within given range
        
        
        long[] dpl, dpr;
        List<Integer>[] adj;
        int[] l,r;
        public void solve(InputReader in, PrintWriter out) {
            int t= in.nextInt();
            while(t-->0){
                int n= in.nextInt();
                l= new int[n];
                r= new int[n];
                adj= new List[n];
                for(int i=0;i<n;i++){
                    l[i]= in.nextInt();
                    r[i]= in.nextInt();
                }
                for(int i=0;i<n;i++) adj[i]= new ArrayList<>();
                for(int i=0;i<n-1;i++){
                    int u= in.nextInt()-1, v= in.nextInt()-1;
                    adj[u].add(v); adj[v].add(u);
                }
                dpl= new long[n]; dpr= new long[n];
                dfs(0,-1);

                out.println(Math.max(dpl[0],dpr[0]));
                
            }
        }

        public void dfs(int u, int p){
            long suml=0, sumr=0;
            for(int v: adj[u] ){
                if(v==p) continue;
                dfs(v,u);
                suml+= Math.max((long)Math.abs((long)l[u]-(long)l[v])+dpl[v], (long)Math.abs((long)l[u]-(long)r[v])+dpr[v]);
                sumr+= Math.max((long)Math.abs((long)r[u]-(long)l[v])+dpl[v],(long) Math.abs((long)r[u]-(long)r[v])+dpr[v]);
            }
            dpl[u]= suml; dpr[u]= sumr;
        }

        // public static class compareL implements Comparator<Tuple>{
        //     @Override
        //     public int compare(Tuple t1, Tuple t2) {
        //         return t2.l - t1.l;
        //     }
        // }
        
        

    }

    // static class Data{
    //     List<Integer> arr;
    //     BigInteger prod=new BigInteger(-1);
    //     public Data(List<Integer> arr, BigInteger prod) {
    //         this.arr= arr;
    //         this.prod= prod;
    //     }
    // }

    

    
    // fast input reader class;
    
}
